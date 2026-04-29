"""
Baselines: naive sazonal, média móvel sazonal, SARIMA (statsforecast), LightGBM.
Avaliados em rolling origin com horizontes (1, 3, 6, 12).
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, SeasonalWindowAverage

from src.eval.metrics import evaluate
from src.utils.paths import DISEASES, PROCESSED, PROJECT_ROOT
from src.utils.splits import DEFAULT_HORIZONS, DEFAULT_ORIGINS, Split, apply_split, rolling_origin


def _panel_to_nixtla(panel: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return (
        panel[["cd_mun", "date", target_col]]
        .rename(columns={"cd_mun": "unique_id", "date": "ds", target_col: "y"})
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def run_stats_baselines(
    panel: pd.DataFrame,
    target_col: str,
    split: Split,
) -> pd.DataFrame:
    train_long = _panel_to_nixtla(panel[panel["date"] <= split.train_end], target_col)
    h = max(split.horizons)
    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=12, alias="seasonal_naive"),
            SeasonalWindowAverage(season_length=12, window_size=3, alias="seasonal_ma3"),
            AutoARIMA(season_length=12, alias="sarima"),
        ],
        freq="MS",
        n_jobs=-1,
    )
    fcst = sf.forecast(df=train_long, h=h)
    fcst = fcst.rename(columns={"unique_id": "cd_mun", "ds": "date"})
    # merge com ground truth
    truth = panel[["cd_mun", "date", target_col]].rename(columns={target_col: "y_true"})
    out = fcst.merge(truth, on=["cd_mun", "date"], how="left")
    return out


def _lag_features(panel: pd.DataFrame, target_col: str, lags=(1, 2, 3, 6, 12, 13), rollings=(3, 6, 12)) -> pd.DataFrame:
    df = panel.sort_values(["cd_mun", "date"]).copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby("cd_mun")[target_col].shift(lag)
    for w in rollings:
        df[f"{target_col}_roll{w}"] = df.groupby("cd_mun")[target_col].shift(1).rolling(w).mean().reset_index(level=0, drop=True)
    # lags de clima
    for c in ("evapot", "precip", "temp_min", "temp_max", "umid"):
        if c in df.columns:
            for lag in (1, 3, 12):
                df[f"{c}_lag{lag}"] = df.groupby("cd_mun")[c].shift(lag)
    return df


def run_lightgbm(
    panel: pd.DataFrame,
    target_col: str,
    split: Split,
    use_count: bool = True,
) -> pd.DataFrame:
    """LightGBM Poisson/Tweedie para contagem; converte taxa fora se necessário."""
    feat = _lag_features(panel, target_col).copy()
    feat["mun_id"] = feat["cd_mun"].astype("category")
    feat["month_of_year"] = feat["month_of_year"].astype("category")
    categorical = ["mun_id", "month_of_year"]
    id_cols = ["cd_mun", "date", "nm_mun"]
    target = target_col
    drop = [target] + [c for c in feat.columns if c.startswith(("tx_", "n_")) and c != target]
    feature_cols = [c for c in feat.columns if c not in id_cols + drop]

    train = feat[feat["date"] <= split.train_end].dropna(subset=feature_cols + [target])
    max_h = max(split.horizons)
    test_end = split.train_end + pd.DateOffset(months=max_h)
    test = feat[(feat["date"] > split.train_end) & (feat["date"] <= test_end)]

    objective = "poisson" if use_count else "tweedie"
    params = dict(
        objective=objective,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
    )
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.3

    model = lgb.LGBMRegressor(n_estimators=400, **params)
    model.fit(train[feature_cols], train[target], categorical_feature=categorical)

    pred = model.predict(test[feature_cols])  # lightgbm handles NaN nativo
    out = test[id_cols + [target]].copy()
    out["yhat_lgbm"] = pred
    out = out.rename(columns={target: "y_true"})
    return out


def run_catboost(
    panel: pd.DataFrame,
    target_col: str,
    split: Split,
    use_count: bool = True,
) -> pd.DataFrame:
    feat = _lag_features(panel, target_col).copy()
    feat["mun_id"] = feat["cd_mun"].astype(str)
    feat["month_of_year"] = feat["month_of_year"].astype(int).astype(str)
    categorical = ["mun_id", "month_of_year"]
    id_cols = ["cd_mun", "date", "nm_mun"]
    target = target_col
    drop = [target] + [c for c in feat.columns if c.startswith(("tx_", "n_")) and c != target]
    feature_cols = [c for c in feat.columns if c not in id_cols + drop]

    train = feat[feat["date"] <= split.train_end].dropna(subset=feature_cols + [target])
    max_h = max(split.horizons)
    test_end = split.train_end + pd.DateOffset(months=max_h)
    test = feat[(feat["date"] > split.train_end) & (feat["date"] <= test_end)]

    loss = "Poisson" if use_count else "Tweedie:variance_power=1.3"
    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function=loss,
        verbose=0,
        random_seed=42,
        cat_features=categorical,
    )
    model.fit(train[feature_cols], train[target])
    pred = model.predict(test[feature_cols])
    out = test[id_cols + [target]].copy()
    out["yhat_catboost"] = pred
    out = out.rename(columns={target: "y_true"})
    return out


def run_all_baselines(
    panel: pd.DataFrame,
    diseases: list[str] = None,
    origins=DEFAULT_ORIGINS,
    horizons=DEFAULT_HORIZONS,
    target_kind: str = "count",  # 'count' treinamos no bruto; pós-processamos taxa se necessário
    save_long_path: str | None = None,
) -> pd.DataFrame:
    diseases = diseases or DISEASES
    records = []
    long_rows = []  # predições por (modelo, doença, horizonte, origem, cd_mun, time_idx)

    for d in diseases:
        target_col = f"n_{d}" if target_kind == "count" else f"tx_{d}"
        for split in rolling_origin(origins, horizons):
            # stats baselines
            stats = run_stats_baselines(panel, target_col, split)
            stats["origin"] = split.name
            stats["disease"] = d

            lgbm = run_lightgbm(panel, target_col, split, use_count=(target_kind == "count"))
            lgbm["origin"] = split.name
            lgbm["disease"] = d

            cat = run_catboost(panel, target_col, split, use_count=(target_kind == "count"))
            cat["origin"] = split.name
            cat["disease"] = d

            # computa métricas por horizonte + acumula long
            for h in horizons:
                h_date = split.train_end + pd.DateOffset(months=h)
                t_idx = (h_date.year - 2000) * 12 + (h_date.month - 1)
                stats_h = stats[stats["date"] == h_date]
                for col in ["seasonal_naive", "seasonal_ma3", "sarima"]:
                    if col not in stats_h.columns or stats_h[col].isna().all():
                        continue
                    records.append(evaluate(stats_h["y_true"].values, stats_h[col].values,
                                            name=col, disease=d, horizon=h) | {"origin": split.name})
                    for _, r in stats_h.iterrows():
                        long_rows.append({"model": col, "disease": d, "horizon": h,
                                          "origin": split.name, "cd_mun": str(r["cd_mun"]).zfill(7),
                                          "time_idx": t_idx, "y_true": r["y_true"], "y_pred": r[col]})
                lgbm_h = lgbm[lgbm["date"] == h_date]
                records.append(evaluate(lgbm_h["y_true"].values, lgbm_h["yhat_lgbm"].values,
                                        name="lgbm", disease=d, horizon=h) | {"origin": split.name})
                for _, r in lgbm_h.iterrows():
                    long_rows.append({"model": "lgbm", "disease": d, "horizon": h,
                                      "origin": split.name, "cd_mun": str(r["cd_mun"]).zfill(7),
                                      "time_idx": t_idx, "y_true": r["y_true"], "y_pred": r["yhat_lgbm"]})
                cat_h = cat[cat["date"] == h_date]
                records.append(evaluate(cat_h["y_true"].values, cat_h["yhat_catboost"].values,
                                        name="catboost", disease=d, horizon=h) | {"origin": split.name})
                for _, r in cat_h.iterrows():
                    long_rows.append({"model": "catboost", "disease": d, "horizon": h,
                                      "origin": split.name, "cd_mun": str(r["cd_mun"]).zfill(7),
                                      "time_idx": t_idx, "y_true": r["y_true"], "y_pred": r["yhat_catboost"]})
    if save_long_path:
        pd.DataFrame(long_rows).to_csv(save_long_path, index=False)
    return pd.DataFrame(records)


if __name__ == "__main__":
    panel = pd.read_parquet(PROCESSED / "panel_23munis.parquet")
    print(f"panel: {panel.shape}, munis={panel['cd_mun'].nunique()}")
    long_path = PROJECT_ROOT / "reports" / "baselines_long.csv"
    df = run_all_baselines(panel, save_long_path=str(long_path))
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    out = PROJECT_ROOT / "reports" / "baselines.csv"
    df.to_csv(out, index=False)
    print(f"\nSalvo em {out}")
    print("\n== Média por modelo × horizonte (agregado entre doenças e origens) ==")
    print(df.groupby(["model", "horizon"])[["mae", "rmse", "smape"]].mean().round(3))
    print("\n== Melhor modelo por doença × horizonte (pelo MAE) ==")
    idx = df.groupby(["disease", "horizon"])["mae"].idxmin()
    print(df.loc[idx, ["disease", "horizon", "model", "mae", "rmse"]].to_string(index=False))
