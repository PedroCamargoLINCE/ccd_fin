"""
Baselines: naive sazonal, média móvel sazonal, SARIMA (statsforecast),
LightGBM, CatBoost, XGBoost e Prophet. Avaliados em rolling origin (1,3,6,12).

============================================================================
CORREÇÃO DE VALIDAÇÃO (revisão 2026-07) — vazamento de lag para h>1
============================================================================
Antes: as features de lag eram calculadas na data-ALVO (T+h). Para h>1 isso
usava valores posteriores à origem T (ex.: prever T+3 com lag1 = valor em T+2),
que não existem no momento da previsão -> vazamento (viés otimista, e injusto
contra os deep models já corrigidos).

Agora (modelos de árvore): esquema DIRETO "as-of-origem". Para a origem T e o
horizonte h, a linha de features é a da ORIGEM T (só passado: valor em T e lags
≤ T, clima em T e lags ≤ T), e o alvo é o valor em T+h. Treina-se um modelo por
horizonte. O mês-alvo (determinístico) entra como covariável de sazonalidade.
Resultado: nenhuma feature > T. Para h=1 é equivalente ao esquema anterior.

Os baselines estatísticos (seasonal_naive, seasonal_ma3, SARIMA) e o Prophet
já preveem h passos à frente a partir de train_end nativamente — não tinham o
vazamento e permanecem inalterados na lógica temporal.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive, SeasonalWindowAverage
from xgboost import XGBRegressor

from src.eval.metrics import evaluate
from src.utils.paths import DISEASES, PROCESSED, PROJECT_ROOT
from src.utils.splits import DEFAULT_HORIZONS, DEFAULT_ORIGINS, Split, rolling_origin

CLIMATE_VARS = ["evapot", "precip", "temp_min", "temp_max", "umid"]
SOCIO_VARS = ["populacao", "dens_demog", "ppc", "urbanizacao"]
TARGET_LAGS = (1, 2, 3, 6, 12, 13)
ROLL_WINDOWS = (3, 6, 12)
CLIMATE_LAGS = (0, 1, 3, 12)
_FEAT_SUFFIXES = tuple(
    [f"_l{k}" for k in (0,) + TARGET_LAGS] + [f"_r{w}" for w in ROLL_WINDOWS]
)


def _panel_to_nixtla(panel: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return (
        panel[["cd_mun", "date", target_col]]
        .rename(columns={"cd_mun": "unique_id", "date": "ds", target_col: "y"})
        .sort_values(["unique_id", "ds"]).reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Estatísticos (nativamente h-passos-à-frente a partir de train_end — sem leak)
# --------------------------------------------------------------------------
def run_stats_baselines(panel: pd.DataFrame, target_col: str, split: Split) -> pd.DataFrame:
    train_long = _panel_to_nixtla(panel[panel["date"] <= split.train_end], target_col)
    h = max(split.horizons)
    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=12, alias="seasonal_naive"),
            SeasonalWindowAverage(season_length=12, window_size=3, alias="seasonal_ma3"),
            AutoARIMA(season_length=12, alias="sarima"),
        ],
        freq="MS", n_jobs=-1,
    )
    fcst = sf.forecast(df=train_long, h=h).rename(columns={"unique_id": "cd_mun", "ds": "date"})
    truth = panel[["cd_mun", "date", target_col]].rename(columns={target_col: "y_true"})
    return fcst.merge(truth, on=["cd_mun", "date"], how="left")


def run_prophet(panel: pd.DataFrame, target_col: str, split: Split) -> pd.DataFrame:
    """Prophet univariado por município (sazonalidade anual). h-passos nativo."""
    from prophet import Prophet
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    h = max(split.horizons)
    rows = []
    for cd_mun, g in panel[panel["date"] <= split.train_end].groupby("cd_mun"):
        dfp = g[["date", target_col]].rename(columns={"date": "ds", target_col: "y"}).dropna()
        if len(dfp) < 24:
            continue
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="additive")
        m.fit(dfp)
        fc = m.predict(m.make_future_dataframe(periods=h, freq="MS"))[["ds", "yhat"]].tail(h)
        fc["yhat"] = fc["yhat"].clip(lower=0)
        fc["cd_mun"] = cd_mun
        rows.append(fc)
    if not rows:
        return pd.DataFrame(columns=["cd_mun", "date", "prophet", "y_true"])
    fcst = pd.concat(rows, ignore_index=True).rename(columns={"ds": "date", "yhat": "prophet"})
    truth = panel[["cd_mun", "date", target_col]].rename(columns={target_col: "y_true"})
    return fcst.merge(truth, on=["cd_mun", "date"], how="left")


# --------------------------------------------------------------------------
# Árvores — esquema DIRETO as-of-origem (corrigido, sem vazamento h>1)
# --------------------------------------------------------------------------
def _asof_features(panel: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = panel.sort_values(["cd_mun", "date"]).copy()
    g = df.groupby("cd_mun")[target_col]
    df[f"{target_col}_l0"] = df[target_col]                       # valor na origem (conhecido)
    for k in TARGET_LAGS:
        df[f"{target_col}_l{k}"] = g.shift(k)
    for w in ROLL_WINDOWS:
        df[f"{target_col}_r{w}"] = g.rolling(w).mean().reset_index(level=0, drop=True)
    for c in CLIMATE_VARS:
        if c in df.columns:
            gc = df.groupby("cd_mun")[c]
            for k in CLIMATE_LAGS:
                df[f"{c}_l{k}"] = df[c] if k == 0 else gc.shift(k)
    return df


def _feature_cols(feat: pd.DataFrame, target_col: str) -> list[str]:
    tgt = [c for c in feat.columns if c.startswith(target_col + "_") and c.endswith(_FEAT_SUFFIXES)]
    clim = [c for c in feat.columns if any(c.startswith(cv + "_l") for cv in CLIMATE_VARS)]
    socio = [c for c in SOCIO_VARS if c in feat.columns]
    return tgt + clim + socio


def _tree_asof(panel, target_col, split, fit_predict, seed) -> pd.DataFrame:
    """Loop de horizontes as-of-origem; retorna long
    [horizon, cd_mun, nm_mun, time_idx, y_true, y_pred]."""
    feat = _asof_features(panel, target_col)
    base_cols = _feature_cols(feat, target_col)
    out = []
    for h in split.horizons:
        f = feat.copy()
        f["_ytgt"] = f.groupby("cd_mun")[target_col].shift(-h)
        f["_tgtdate"] = f["date"] + pd.DateOffset(months=h)
        f["tgt_month"] = f["_tgtdate"].dt.month
        f["mun_id"] = f["cd_mun"]
        cols = base_cols + ["tgt_month", "mun_id"]
        train = f[f["_tgtdate"] <= split.train_end].dropna(subset=base_cols + ["_ytgt"])
        test = f[f["date"] == split.train_end].copy()
        if len(train) == 0 or len(test) == 0:
            continue
        preds = fit_predict(train, test, cols, "_ytgt", seed)
        t0 = int(test["time_idx"].iloc[0])
        for (_, r), yhat in zip(test.iterrows(), preds):
            out.append({"horizon": h, "cd_mun": r["cd_mun"], "nm_mun": r.get("nm_mun"),
                        "time_idx": t0 + h, "y_true": float(r["_ytgt"]),
                        "y_pred": float(max(yhat, 0.0))})
    return pd.DataFrame(out)


def _fp_lgbm(train, test, cols, ycol, seed):
    cat = ["tgt_month", "mun_id"]
    tr = train.copy(); te = test.copy()
    for c in cat:
        tr[c] = tr[c].astype("category"); te[c] = te[c].astype("category")
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
                          bagging_freq=5, objective="poisson", random_state=seed, verbose=-1)
    m.fit(tr[cols], tr[ycol], categorical_feature=cat)
    return m.predict(te[cols])


def _fp_catboost(train, test, cols, ycol, seed):
    cat = ["tgt_month", "mun_id"]
    tr = train.copy(); te = test.copy()
    for c in cat:
        tr[c] = tr[c].astype(str); te[c] = te[c].astype(str)
    m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                          loss_function="Poisson", verbose=0, random_seed=seed, cat_features=cat)
    m.fit(tr[cols], tr[ycol]); return m.predict(te[cols])


def _fp_xgboost(train, test, cols, ycol, seed):
    cat = ["tgt_month", "mun_id"]
    tr = train.copy(); te = test.copy()
    for c in cat:
        tr[c] = tr[c].astype("category"); te[c] = te[c].astype("category")
    m = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                     objective="count:poisson", tree_method="hist", enable_categorical=True,
                     random_state=seed, n_jobs=-1)
    m.fit(tr[cols], tr[ycol]); return m.predict(te[cols])


_TREE_FP = {"lgbm": _fp_lgbm, "catboost": _fp_catboost, "xgboost": _fp_xgboost}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def run_all_baselines(
    panel: pd.DataFrame,
    diseases: list[str] = None,
    origins=DEFAULT_ORIGINS,
    horizons=DEFAULT_HORIZONS,
    target_kind: str = "count",
    seed: int = 42,
    include_prophet: bool = True,
    save_long_path: str | None = None,
) -> pd.DataFrame:
    diseases = diseases or DISEASES
    records, long_rows = [], []

    def _add_long(model, d, h, origin, cd, t_idx, yt, yp):
        long_rows.append({"model": model, "disease": d, "horizon": h, "origin": origin,
                          "cd_mun": str(cd).zfill(7), "time_idx": int(t_idx),
                          "y_true": yt, "y_pred": yp})

    for d in diseases:
        target_col = f"n_{d}" if target_kind == "count" else f"tx_{d}"
        for split in rolling_origin(origins, horizons):
            stats = run_stats_baselines(panel, target_col, split)
            proph = run_prophet(panel, target_col, split) if include_prophet else None
            tree_long = {m: _tree_asof(panel, target_col, split, _TREE_FP[m], seed)
                         for m in ("lgbm", "catboost", "xgboost")}

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
                        _add_long(col, d, h, split.name, r["cd_mun"], t_idx, r["y_true"], r[col])

                for m in ("lgbm", "catboost", "xgboost"):
                    th = tree_long[m][tree_long[m]["horizon"] == h]
                    if len(th) == 0:
                        continue
                    records.append(evaluate(th["y_true"].values, th["y_pred"].values,
                                            name=m, disease=d, horizon=h) | {"origin": split.name})
                    for _, r in th.iterrows():
                        _add_long(m, d, h, split.name, r["cd_mun"], r["time_idx"], r["y_true"], r["y_pred"])

                if proph is not None:
                    ph = proph[proph["date"] == h_date]
                    if len(ph) and not ph["prophet"].isna().all():
                        records.append(evaluate(ph["y_true"].values, ph["prophet"].values,
                                                name="prophet", disease=d, horizon=h) | {"origin": split.name})
                        for _, r in ph.iterrows():
                            _add_long("prophet", d, h, split.name, r["cd_mun"], t_idx, r["y_true"], r["prophet"])

    if save_long_path:
        pd.DataFrame(long_rows).to_csv(save_long_path, index=False)
    return pd.DataFrame(records)


if __name__ == "__main__":
    panel = pd.read_parquet(PROCESSED / "panel_23munis.parquet")
    print(f"panel: {panel.shape}, munis={panel['cd_mun'].nunique()}")
    long_path = PROJECT_ROOT / "reports" / "baselines_long.csv"
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    df = run_all_baselines(panel, save_long_path=str(long_path))
    out = PROJECT_ROOT / "reports" / "baselines.csv"
    df.to_csv(out, index=False)
    print(f"\nSalvo em {out}")
    print("\n== Média por modelo × horizonte ==")
    print(df.groupby(["model", "horizon"])[["mae", "rmse", "smape"]].mean().round(3))
