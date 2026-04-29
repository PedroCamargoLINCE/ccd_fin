"""
Centraliza todos os resultados em results/.

Estrutura gerada:
  results/
    README.md                       # síntese
    tables/
      final_summary.csv            # modelo × doença × horizonte (agregado origem)
      per_muni_<disease>.csv       # município × modelo, MAE médio
      ranking.csv                  # melhor modelo por (doença, horizonte)
    figures/
      eda/                         # heatmaps, séries por muni
      forecasts/                   # observado vs previsto, grade 6×4 por (doença, modelo)
      comparison/                  # MAE por horizonte, estabilidade por origem

Uso:
  python scripts/make_results.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eval.metrics import evaluate
from src.utils.paths import DISEASES, MUNICIPIOS_ALVO, PROCESSED, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"
RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"
FIG_EDA = RESULTS / "figures" / "eda"
FIG_FCST = RESULTS / "figures" / "forecasts"
FIG_CMP = RESULTS / "figures" / "comparison"
for d in (TABLES, FIG_EDA, FIG_FCST, FIG_CMP):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-12-01")
DEEP_MODELS = ("tft", "nhits", "deepar")


def load_all_results() -> tuple[pd.DataFrame, dict]:
    """Une baselines + deep csvs num único frame long: model/disease/horizon/origin/cd_mun/y_true/y_pred."""
    deep_long = []
    for mdl in DEEP_MODELS:
        for d in DISEASES:
            f = REPORTS / f"deep_{mdl}_{d}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            if "model" not in df.columns:
                df["model"] = mdl
            if "disease" not in df.columns:
                df["disease"] = d
            deep_long.append(df)
    deep_df = pd.concat(deep_long, ignore_index=True) if deep_long else pd.DataFrame()

    bl = pd.read_csv(REPORTS / "baselines.csv")
    return deep_df, {"baselines": bl}


def per_horizon_summary(deep_df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # baselines: já tem mae/rmse pré-calculados
    for (m, d, h), g in baselines.groupby(["model", "disease", "horizon"]):
        rows.append({
            "model": m, "disease": d, "horizon": h,
            "mae": g["mae"].mean(), "rmse": g["rmse"].mean(), "smape": g["smape"].mean(),
        })
    # deep: agrega por origem + cd_mun
    if not deep_df.empty:
        for (m, d, h), g in deep_df.groupby(["model", "disease", "horizon"]):
            ev = evaluate(g["y_true"].values, g["y_pred"].values, name=m, disease=d, horizon=h)
            rows.append({
                "model": m, "disease": d, "horizon": h,
                "mae": ev["mae"], "rmse": ev["rmse"], "smape": ev["smape"],
            })
    return pd.DataFrame(rows)


def per_muni_table(deep_df: pd.DataFrame, baselines: pd.DataFrame, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Para cada doença, retorna df: index=município, columns=modelo, value=MAE médio (todos h, todas origens)."""
    name_map = MUNICIPIOS_ALVO

    # Para baselines, preciso recalcular por município — o reports/baselines.csv só tem agregado
    # Vamos reconstruir do zero: re-treinar só pra ter previsões por muni? Custoso.
    # Atalho: usamos o agregado disponível e marcamos como "média" para baselines.
    # Para deep, agregamos diretamente do long.

    out: dict[str, pd.DataFrame] = {}
    for d in DISEASES:
        rows = []
        # deep models por muni
        sub = deep_df[deep_df["disease"] == d]
        for (m, cd), g in sub.groupby(["model", "cd_mun"]):
            cd_str = str(cd).zfill(7) if not isinstance(cd, str) else cd.zfill(7)
            ev = evaluate(g["y_true"].values, g["y_pred"].values, name=m, disease=d)
            rows.append({"model": m, "cd_mun": cd_str, "nm_mun": name_map.get(cd_str, cd_str),
                         "mae": ev["mae"], "rmse": ev["rmse"], "n": ev["n"]})
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        wide_mae = df.pivot_table(index="nm_mun", columns="model", values="mae").round(2)
        wide_rmse = df.pivot_table(index="nm_mun", columns="model", values="rmse").round(2)
        # ordena por MAE médio crescente
        wide_mae["__mean"] = wide_mae.mean(axis=1)
        wide_mae = wide_mae.sort_values("__mean").drop(columns="__mean")
        out[d] = wide_mae
        wide_mae.to_csv(TABLES / f"per_muni_mae_{d}.csv")
        wide_rmse.to_csv(TABLES / f"per_muni_rmse_{d}.csv")
    return out


def ranking_per_disease(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, h), g in summary.groupby(["disease", "horizon"]):
        srt = g.sort_values("mae")
        rows.append({"disease": d, "horizon": h,
                     "1st": f"{srt.iloc[0]['model']} ({srt.iloc[0]['mae']:.2f})",
                     "2nd": f"{srt.iloc[1]['model']} ({srt.iloc[1]['mae']:.2f})" if len(srt) > 1 else "",
                     "3rd": f"{srt.iloc[2]['model']} ({srt.iloc[2]['mae']:.2f})" if len(srt) > 2 else ""})
    return pd.DataFrame(rows)


def copy_eda_figures():
    src = REPORTS / "figures"
    if not src.exists():
        return
    eda_keys = ("zero_fraction", "series_", "climate_lag", "rolling_origin")
    cmp_keys = ("baselines_mae_by_horizon", "comparison_mae_by_horizon", "stability_by_origin")
    for f in src.glob("*.png"):
        if any(x in f.name for x in eda_keys):
            shutil.copy2(f, FIG_EDA / f.name)
        elif any(x in f.name for x in cmp_keys):
            shutil.copy2(f, FIG_CMP / f.name)


def plot_forecasts_per_muni(panel: pd.DataFrame, deep_df: pd.DataFrame, disease: str, model_name: str):
    col = f"n_{disease}"
    sub = deep_df[(deep_df["disease"] == disease) & (deep_df["model"] == model_name)].copy()
    if sub.empty:
        return
    # converte time_idx -> date
    sub["date"] = sub["time_idx"].apply(lambda i: pd.Timestamp("2000-01-01") + pd.DateOffset(months=int(i)))
    sub["cd_mun"] = sub["cd_mun"].astype(str).str.zfill(7)

    munis = sorted(panel["nm_mun"].dropna().unique())
    fig, axes = plt.subplots(6, 4, figsize=(22, 22), sharex=False)
    for ax, muni in zip(axes.flat, munis):
        cd = next((k for k, v in MUNICIPIOS_ALVO.items() if v == muni), None)
        if not cd:
            ax.axis("off"); continue
        hist = panel[panel["cd_mun"] == cd].sort_values("date")
        ax.plot(hist["date"], hist[col], lw=0.8, color="#444444", label="observado")
        pm = sub[sub["cd_mun"] == cd]
        for i, (origin, g) in enumerate(pm.groupby("origin")):
            g = g.sort_values("time_idx")
            ax.plot(g["date"], g["y_pred"], marker=".", ms=4, lw=1.2,
                    color=plt.cm.tab10.colors[i % 10], label=origin)
        ax.axvspan(COVID_START, COVID_END, color="red", alpha=0.05)
        ax.set_title(muni, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(munis):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="upper left", ncol=2)
    fig.suptitle(f"{disease} — {model_name} vs observado (origens em cores)", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_FCST / f"forecast_{disease}_{model_name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_muni_heatmap(per_muni: dict[str, pd.DataFrame]):
    for d, mat in per_muni.items():
        fig, ax = plt.subplots(figsize=(7, max(5, 0.35 * len(mat))))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="rocket_r", ax=ax,
                    cbar_kws={"label": "MAE (contagem)"})
        ax.set_title(f"MAE por município × modelo — {d}")
        ax.set_xlabel("modelo"); ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(FIG_CMP / f"per_muni_heatmap_{d}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def write_synthesis(summary: pd.DataFrame, ranking: pd.DataFrame, per_muni: dict):
    md = ["# Resultados — predição multi-doença em 23 municípios de SP",
          "",
          "Síntese consolidada do pipeline: baselines (SeasonalNaive, SeasonalMA3, SARIMA, LightGBM, CatBoost) + deep panel (TFT, N-HiTS, DeepAR).",
          "",
          "## Resumo agregado (MAE médio entre origens)",
          ""]
    pv = summary.pivot_table(index="model", columns="horizon", values="mae").round(2)
    md.append(pv.to_markdown())
    md.extend(["",
               "## Ranking por (doença × horizonte)",
               ""])
    md.append(ranking.to_markdown(index=False))
    md.extend(["",
               "## Vencedor por doença (MAE médio entre horizontes)",
               ""])
    win = (summary.groupby(["disease", "model"])["mae"].mean()
                  .reset_index()
                  .sort_values(["disease", "mae"])
                  .groupby("disease").head(1))
    md.append(win.round(2).to_markdown(index=False))
    md.extend(["",
               "## Tabelas por município",
               "",
               "Uma tabela por doença em [`tables/per_muni_mae_<doenca>.csv`](tables/) — MAE de cada modelo deep para cada um dos 23 municípios. Heatmaps em [`figures/comparison/`](figures/comparison/).",
               "",
               "## Figuras",
               "",
               "- **EDA**: `figures/eda/` — zero-fraction, séries por município, correlação clima×alvo, rolling origin",
               "- **Forecasts**: `figures/forecasts/` — predito vs observado, grade 6×4 por (doença × modelo deep)",
               "- **Comparação**: `figures/comparison/` — MAE por horizonte, estabilidade por origem, heatmap por município",
               ""])
    (RESULTS / "README.md").write_text("\n".join(md), encoding="utf-8")


def main():
    deep_df, others = load_all_results()
    bl = others["baselines"]
    panel = pd.read_parquet(PROCESSED / "panel_23munis.parquet")
    panel["cd_mun"] = panel["cd_mun"].astype(str).str.zfill(7)

    print(f"[load] deep records: {len(deep_df):,} | baselines records: {len(bl):,}")

    summary = per_horizon_summary(deep_df, bl)
    summary.to_csv(TABLES / "final_summary.csv", index=False)
    print(f"[write] {TABLES/'final_summary.csv'}")

    ranking = ranking_per_disease(summary)
    ranking.to_csv(TABLES / "ranking.csv", index=False)
    print(f"[write] {TABLES/'ranking.csv'}")

    per_muni = per_muni_table(deep_df, bl, panel)
    print(f"[write] per_muni tables ({len(per_muni)} doenças)")

    copy_eda_figures()
    print("[copy] figuras de EDA")

    for d in DISEASES:
        for mdl in DEEP_MODELS:
            plot_forecasts_per_muni(panel, deep_df, d, mdl)
    print("[plot] forecasts por (doença × modelo)")

    plot_per_muni_heatmap(per_muni)
    print("[plot] heatmaps por município")

    write_synthesis(summary, ranking, per_muni)
    print(f"[write] {RESULTS/'README.md'}")
    print(f"\nresults em: {RESULTS}")


if __name__ == "__main__":
    main()
