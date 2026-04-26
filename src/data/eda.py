"""
EDA crítica: decide a estratégia de loss (NB / ZINB / Tweedie / hurdle) com
base na fração de zeros observada por (doença × município). Também produz
relatório de missingness, sazonalidade e correlação lag clima × alvo.

Saída em reports/eda/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.build_panel import build_panel
from src.utils.paths import DISEASES, PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports" / "eda"
REPORTS.mkdir(parents=True, exist_ok=True)


def zero_fraction_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in DISEASES:
        col = f"n_{d}"
        if col not in panel.columns:
            continue
        for cd, g in panel.groupby("cd_mun"):
            v = g[col].dropna()
            rows.append({
                "doenca": d,
                "cd_mun": cd,
                "nm_mun": g["nm_mun"].iloc[0],
                "n_obs": len(v),
                "zero_frac": (v == 0).mean() if len(v) else np.nan,
                "mean": v.mean() if len(v) else np.nan,
                "max": v.max() if len(v) else np.nan,
                "var_mean_ratio": (v.var() / v.mean()) if len(v) and v.mean() > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def recommend_loss(zf: pd.DataFrame) -> dict[str, str]:
    """Heurística: escolhe loss por doença com base na fração de zeros + sobredispersão."""
    rec = {}
    for d, g in zf.groupby("doenca"):
        max_zf = g["zero_frac"].max()
        med_zf = g["zero_frac"].median()
        vm = g["var_mean_ratio"].median()
        if med_zf >= 0.5 or max_zf >= 0.8:
            rec[d] = "ZINB (zero-inflated)"
        elif vm > 1.5:
            rec[d] = "NegativeBinomial"
        else:
            rec[d] = "Poisson"
    return rec


def plot_zero_heatmap(zf: pd.DataFrame, out: Path):
    piv = zf.pivot(index="nm_mun", columns="doenca", values="zero_frac")
    plt.figure(figsize=(8, 8))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1, cbar_kws={"label": "% zeros"})
    plt.title("Fração de meses com 0 casos — (município × doença)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_series(panel: pd.DataFrame, disease: str, out: Path):
    col_tx = f"tx_{disease}"
    if col_tx not in panel.columns:
        return
    fig, axes = plt.subplots(6, 4, figsize=(20, 18), sharex=True)
    munis = panel["nm_mun"].dropna().unique()
    for ax, muni in zip(axes.flat, munis):
        g = panel[panel["nm_mun"] == muni].sort_values("date")
        ax.plot(g["date"], g[col_tx], lw=0.8)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"), color="red", alpha=0.1)
        ax.set_title(muni, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(munis):]:
        ax.axis("off")
    fig.suptitle(f"Taxa mensal — {disease} (faixa vermelha = período COVID)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def missingness_table(panel: pd.DataFrame) -> pd.DataFrame:
    m = panel.isna().mean().rename("pct_missing").to_frame()
    m["n_missing"] = panel.isna().sum()
    return m.sort_values("pct_missing", ascending=False)


def climate_lag_correlation(panel: pd.DataFrame, disease: str, max_lag: int = 6) -> pd.DataFrame:
    """Spearman entre clima_t-k e taxa_t, média entre municípios."""
    climate_vars = ["evapot", "precip", "temp_min", "temp_max", "umid"]
    col_tx = f"tx_{disease}"
    rows = []
    for cv in climate_vars:
        if cv not in panel.columns:
            continue
        for lag in range(0, max_lag + 1):
            corrs = []
            for _, g in panel.groupby("cd_mun"):
                g = g.sort_values("date")
                lagged = g[cv].shift(lag)
                c = g[col_tx].corr(lagged, method="spearman")
                if pd.notna(c):
                    corrs.append(c)
            rows.append({"clima": cv, "lag": lag, "spearman_mean": np.mean(corrs) if corrs else np.nan})
    return pd.DataFrame(rows)


def run():
    panel = build_panel()
    panel.to_parquet(REPORTS / "panel_snapshot.parquet", index=False)

    print("== missingness ==")
    miss = missingness_table(panel)
    miss.to_csv(REPORTS / "missingness.csv")
    print(miss.head(15))

    print("\n== zero fraction ==")
    zf = zero_fraction_table(panel)
    zf.to_csv(REPORTS / "zero_fraction.csv", index=False)
    plot_zero_heatmap(zf, REPORTS / "zero_fraction_heatmap.png")
    summary = zf.groupby("doenca").agg(
        zf_median=("zero_frac", "median"),
        zf_max=("zero_frac", "max"),
        vm_median=("var_mean_ratio", "median"),
    )
    print(summary)

    rec = recommend_loss(zf)
    print("\n== recomendação de loss por doença ==")
    for d, r in rec.items():
        print(f"  {d}: {r}")
    pd.Series(rec).to_csv(REPORTS / "loss_recommendation.csv")

    print("\n== série temporal por doença ==")
    for d in DISEASES:
        plot_series(panel, d, REPORTS / f"series_{d}.png")
        print(f"  saved series_{d}.png")

    print("\n== correlação lag clima × taxa ==")
    corrs = []
    for d in DISEASES:
        c = climate_lag_correlation(panel, d)
        c["doenca"] = d
        corrs.append(c)
    full = pd.concat(corrs, ignore_index=True)
    full.to_csv(REPORTS / "climate_lag_corr.csv", index=False)
    print(full.pivot_table(index=["doenca", "clima"], columns="lag", values="spearman_mean").round(3))

    print(f"\nRelatório salvo em {REPORTS}")


if __name__ == "__main__":
    run()
