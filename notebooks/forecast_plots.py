"""
Plots de forecast por (município × doença) com TODOS os 8 modelos sobrepostos.

Granularidade:
- Uma figura por par (cd_mun × doença)
- 4 subplots (um por origem do rolling window): mostra ~36 meses de contexto
  observado + 12 meses de horizonte com a previsão de cada modelo
- Total: 5 doenças × 23 munis = 115 figuras

Saídas em `results/forecasts_detailed/<doenca>/<muni>.png`.
"""
# %% [markdown]
# # Forecasts detalhados por município × doença
#
# Uma figura por par. Em cada figura, 4 subplots (um por origem do rolling
# origin) com **todos os 8 modelos** sobrepostos. Inclui baselines (SARIMA,
# LightGBM, CatBoost) que estavam ausentes nos plots anteriores.

# %%
import sys, os
from pathlib import Path
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def _detect_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "src" / "utils" / "paths.py").exists():
            return p
    return Path.cwd()
PROJECT_ROOT = _detect_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 110

from src.utils.paths import DISEASES, MUNICIPIOS_ALVO, PROCESSED

REPORTS = PROJECT_ROOT / "reports"
OUT = PROJECT_ROOT / "results" / "forecasts_detailed"
OUT.mkdir(parents=True, exist_ok=True)
for d in DISEASES:
    (OUT / d).mkdir(exist_ok=True)

DEEP_MODELS = ["tft", "nhits", "deepar"]
BASELINE_MODELS = ["seasonal_naive", "seasonal_ma3", "sarima", "lgbm", "catboost"]

ORIGINS = ["origin_201912", "origin_202012", "origin_202112", "origin_202212"]
ORIGIN_DATES = {
    "origin_201912": pd.Timestamp("2019-12-01"),
    "origin_202012": pd.Timestamp("2020-12-01"),
    "origin_202112": pd.Timestamp("2021-12-01"),
    "origin_202212": pd.Timestamp("2022-12-01"),
}
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-12-01")

# paleta consistente: tons frios pros baselines, quentes pros deep
PALETTE = {
    "seasonal_naive": "#a9c4d6",
    "seasonal_ma3":   "#7aa3bf",
    "sarima":         "#3c6e91",
    "lgbm":           "#1f4e79",
    "catboost":       "#0d2c47",
    "tft":            "#d62728",
    "nhits":          "#ff7f0e",
    "deepar":         "#e377c2",
}

# %% [markdown]
# ## 1. Garantir predições e consolidar

# %%
def ensure_baselines_long():
    bl_long = REPORTS / "baselines_long.csv"
    if bl_long.exists():
        print(f"[ok] baselines_long.csv já existe ({bl_long.stat().st_size//1024} KB)")
        return
    print("[run] reports/baselines_long.csv não existe — rodando baselines (~5-10 min)...")
    import pandas as _pd
    from src.models.baselines import run_all_baselines
    panel = _pd.read_parquet(PROCESSED / "panel_23munis.parquet")
    run_all_baselines(panel, save_long_path=str(bl_long))
    print(f"[done] salvo em {bl_long}")

def load_all_predictions() -> pd.DataFrame:
    frames = []
    for mdl in DEEP_MODELS:
        for d in DISEASES:
            f = REPORTS / f"deep_{mdl}_{d}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df["model"] = mdl; df["disease"] = d
            df["cd_mun"] = df["cd_mun"].astype(str).str.zfill(7)
            frames.append(df[["model","disease","horizon","origin","cd_mun","time_idx","y_true","y_pred"]])
    bl = pd.read_csv(REPORTS / "baselines_long.csv")
    bl["cd_mun"] = bl["cd_mun"].astype(str).str.zfill(7)
    frames.append(bl[["model","disease","horizon","origin","cd_mun","time_idx","y_true","y_pred"]])
    return pd.concat(frames, ignore_index=True)

ensure_baselines_long()
preds = load_all_predictions()
preds["date"] = preds["time_idx"].apply(lambda i: pd.Timestamp("2000-01-01") + pd.DateOffset(months=int(i)))
print(f"total predições: {len(preds):,}")
print(f"modelos: {sorted(preds['model'].unique())}")

# %% [markdown]
# ## 2. Painel observado (para o contexto histórico)

# %%
panel = pd.read_parquet(PROCESSED / "panel_23munis.parquet")
panel["cd_mun"] = panel["cd_mun"].astype(str).str.zfill(7)
panel = panel.sort_values(["cd_mun", "date"])

# %% [markdown]
# ## 3. Função de plot por (município × doença)

# %%
def plot_one(disease: str, cd_mun: str, nm_mun: str, context_months: int = 36):
    col = f"n_{disease}"
    hist = panel[panel["cd_mun"] == cd_mun][["date", col]].rename(columns={col: "y"})

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharey=True)
    p_muni = preds[(preds["disease"] == disease) & (preds["cd_mun"] == cd_mun)]

    for ax, origin in zip(axes, ORIGINS):
        origin_date = ORIGIN_DATES[origin]
        max_h_date = origin_date + pd.DateOffset(months=12)
        ctx_start = origin_date - pd.DateOffset(months=context_months)

        # observado (preto, grosso)
        ctx = hist[(hist["date"] >= ctx_start) & (hist["date"] <= max_h_date)]
        ax.plot(ctx["date"], ctx["y"], color="black", lw=1.6, label="observado", zorder=10)

        # vertical line marcando origem
        ax.axvline(origin_date, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.axvspan(COVID_START, COVID_END, color="red", alpha=0.05)

        # cada modelo
        sub = p_muni[p_muni["origin"] == origin]
        for m in sub["model"].unique():
            g = sub[sub["model"] == m].sort_values("time_idx")
            color = PALETTE.get(m, "#888888")
            ax.plot(g["date"], g["y_pred"], marker=".", ms=5, lw=1.2, color=color, label=m, alpha=0.9)

        ax.set_title(f"{nm_mun} — {disease} — {origin.replace('origin_','')}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("contagem")

    # legenda única embaixo
    handles = [Line2D([0],[0], color="black", lw=1.6, label="observado")] + \
              [Line2D([0],[0], color=PALETTE[m], marker=".", lw=1.2, label=m) for m in BASELINE_MODELS + DEEP_MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{nm_mun} — {disease}: forecast de cada modelo por origem", y=1.00, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    out_path = OUT / disease / f"{cd_mun}_{nm_mun.replace(' ','_').replace('ã','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ç','c').replace('â','a').replace('ô','o')}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path

# %% [markdown]
# ## 4. Gerar tudo (5 × 23 = 115 figuras)

# %%
total = 0
for d in DISEASES:
    for cd, nm in MUNICIPIOS_ALVO.items():
        plot_one(d, cd, nm)
        total += 1
    print(f"[done] {d}: 23 munis salvos em {OUT/d}")

print(f"\nTotal: {total} figuras em {OUT}")

# %% [markdown]
# ## 5. Plus: comparação modelo-a-modelo (todos os munis num único arquivo, por modelo × doença)
#
# Grade 4×6 = 24 painéis (23 munis + 1 vazio), por (modelo × doença).
# Total: 8 modelos × 5 doenças = 40 figuras.

# %%
GRID_OUT = PROJECT_ROOT / "results" / "forecasts_grid"
GRID_OUT.mkdir(exist_ok=True)
ALL_MODELS = BASELINE_MODELS + DEEP_MODELS

def plot_grid(disease: str, model: str):
    col = f"n_{disease}"
    sub = preds[(preds["disease"] == disease) & (preds["model"] == model)].copy()
    if sub.empty:
        return
    munis = sorted(panel["nm_mun"].dropna().unique())
    fig, axes = plt.subplots(4, 6, figsize=(26, 14), sharex=False)
    for ax, muni in zip(axes.flat, munis):
        cd = next((k for k, v in MUNICIPIOS_ALVO.items() if v == muni), None)
        if not cd:
            ax.axis("off"); continue
        hist = panel[panel["cd_mun"] == cd]
        ax.plot(hist["date"], hist[col], color="black", lw=0.7, label="observado")
        pm = sub[sub["cd_mun"] == cd]
        for i, (origin, g) in enumerate(pm.groupby("origin")):
            g = g.sort_values("time_idx")
            ax.plot(g["date"], g["y_pred"], marker=".", ms=3, lw=1.0,
                    color=plt.cm.tab10.colors[i % 10], label=origin.replace("origin_",""), alpha=0.9)
        ax.axvspan(COVID_START, COVID_END, color="red", alpha=0.05)
        ax.set_title(muni, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(munis):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="upper left", ncol=2)
    fig.suptitle(f"{disease} — {model}: forecast por município (4 origens em cores)", y=1.00, fontsize=12)
    fig.tight_layout()
    fig.savefig(GRID_OUT / f"{disease}_{model}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

count = 0
for d in DISEASES:
    for m in ALL_MODELS:
        plot_grid(d, m)
        count += 1
print(f"[done] {count} grades modelo×doença em {GRID_OUT}")
