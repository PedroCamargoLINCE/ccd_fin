"""
Ensemble dos top-3 modelos por (doença × horizonte) com **split temporal honesto**:
- **Validação** (escolhe os top-3 e os pesos): origens 2019-12 e 2020-12
- **Teste** (avalia o ensemble): origens 2021-12 e 2022-12

Estratégias comparadas:
- `top3_mean`     — média simples dos 3 melhores na validação
- `top3_inv_mae`  — média ponderada por 1/MAE (pesos da validação)
- `all_mean`      — sanidade: média de todos os 8 modelos

Saídas em `results/ensemble/`.
"""
# %% [markdown]
# # Ensemble (top-3 por doença × horizonte)
#
# Combina baselines + deep panel para reduzir variância. Usamos split temporal
# honesto: validação em 2019-2020 (escolhe modelos e pesos), teste em 2021-2022
# (avalia o ensemble).

# %%
import sys, os
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def _detect_root() -> Path:
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
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

from src.utils.paths import DISEASES, MUNICIPIOS_ALVO, PROCESSED
from src.eval.metrics import evaluate

REPORTS = PROJECT_ROOT / "reports"
RESULTS = PROJECT_ROOT / "results"
ENS_DIR = RESULTS / "ensemble"
ENS_DIR.mkdir(parents=True, exist_ok=True)
(ENS_DIR / "figures").mkdir(exist_ok=True)
(ENS_DIR / "tables").mkdir(exist_ok=True)

VAL_ORIGINS = ["origin_201912", "origin_202012"]
TEST_ORIGINS = ["origin_202112", "origin_202212"]
DEEP_MODELS = ["tft", "nhits", "deepar"]

# %% [markdown]
# ## 1. Garantir que as predições existem (deep + baselines long)
#
# - Predições dos modelos deep ficam em `reports/deep_<model>_<doença>.csv` (geradas
#   pelo `train_all.ipynb`).
# - Predições dos baselines em formato long ficam em `reports/baselines_long.csv`.
#   Se faltar, esta célula re-roda os baselines (~5-10 min, sem GPU).

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

def ensure_deep_predictions():
    """Avisa se algum CSV deep estiver faltando — não retreina (precisa GPU)."""
    missing = []
    for mdl in DEEP_MODELS:
        for d in DISEASES:
            f = REPORTS / f"deep_{mdl}_{d}.csv"
            if not f.exists():
                missing.append(f.name)
    if missing:
        print(f"AVISO: {len(missing)} CSVs deep faltando — rode train_all.ipynb antes:")
        for m in missing[:5]:
            print(f"  - {m}")
        if len(missing) > 5:
            print(f"  ... (+{len(missing)-5})")
    else:
        print(f"[ok] todos os {len(DEEP_MODELS)*len(DISEASES)} CSVs deep presentes")

ensure_baselines_long()
ensure_deep_predictions()

# %% [markdown]
# ## 2. Consolidar predições num único frame long

# %%
def load_predictions() -> pd.DataFrame:
    frames = []

    # deep
    for mdl in DEEP_MODELS:
        for d in DISEASES:
            f = REPORTS / f"deep_{mdl}_{d}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            df["model"] = mdl
            df["disease"] = d
            df["cd_mun"] = df["cd_mun"].astype(str).str.zfill(7)
            frames.append(df[["model", "disease", "horizon", "origin", "cd_mun", "time_idx", "y_true", "y_pred"]])

    # baselines (long)
    bl_long = REPORTS / "baselines_long.csv"
    if bl_long.exists():
        bl = pd.read_csv(bl_long)
        bl["cd_mun"] = bl["cd_mun"].astype(str).str.zfill(7)
        frames.append(bl[["model", "disease", "horizon", "origin", "cd_mun", "time_idx", "y_true", "y_pred"]])

    return pd.concat(frames, ignore_index=True)

preds = load_predictions()
print(f"total predições: {len(preds):,}")
print(f"modelos: {sorted(preds['model'].unique().tolist())}")
print(f"origens: {sorted(preds['origin'].unique().tolist())}")

# %% [markdown]
# ## 2. Ranking dos modelos na validação (2019, 2020)

# %%
def model_mae_on_origins(preds, origins):
    sub = preds[preds["origin"].isin(origins)]
    rows = []
    for (m, d, h), g in sub.groupby(["model", "disease", "horizon"]):
        ev = evaluate(g["y_true"].values, g["y_pred"].values, name=m, disease=d, horizon=h)
        rows.append({"model": m, "disease": d, "horizon": h, "mae": ev["mae"], "n": ev["n"]})
    return pd.DataFrame(rows)

val_rank = model_mae_on_origins(preds, VAL_ORIGINS)

def top_k(val_rank, disease, horizon, k=3):
    sub = val_rank[(val_rank["disease"] == disease) & (val_rank["horizon"] == horizon)]
    return sub.sort_values("mae").head(k)

print("Top-3 por (doença × horizonte) — escolhido em 2019-2020:")
for d in DISEASES:
    for h in (1, 3, 6, 12):
        t = top_k(val_rank, d, h)
        names = " | ".join(f"{r['model']} ({r['mae']:.2f})" for _, r in t.iterrows())
        print(f"  {d:12s} h={h:02d}: {names}")

val_rank.to_csv(ENS_DIR / "tables" / "val_ranking.csv", index=False)

# %% [markdown]
# ## 3. Construir as predições do ensemble (no teste)
#
# Para cada (doença × horizonte × cd_mun × time_idx), agrega previsões dos top-3
# usando 3 estratégias.

# %%
def build_ensembles(preds, val_rank):
    test = preds[preds["origin"].isin(TEST_ORIGINS)].copy()

    # pivot: (disease, horizon, cd_mun, time_idx, origin) × model -> y_pred
    pv = test.pivot_table(
        index=["disease", "horizon", "origin", "cd_mun", "time_idx", "y_true"],
        columns="model", values="y_pred",
    ).reset_index()

    rows_top3_mean = []
    rows_top3_invmae = []
    rows_all_mean = []
    all_models = sorted(preds["model"].unique())

    for (d, h), g in pv.groupby(["disease", "horizon"]):
        ranking = val_rank[(val_rank["disease"] == d) & (val_rank["horizon"] == h)].sort_values("mae")
        top3 = ranking.head(3)
        top3_models = top3["model"].tolist()
        top3_mae = top3.set_index("model")["mae"]

        # média simples
        if all(m in g.columns for m in top3_models):
            mean_pred = g[top3_models].mean(axis=1)
            for i, row in g.reset_index().iterrows():
                rows_top3_mean.append({
                    "disease": d, "horizon": h, "origin": row["origin"],
                    "cd_mun": row["cd_mun"], "time_idx": row["time_idx"],
                    "y_true": row["y_true"], "y_pred": mean_pred.iloc[i],
                    "members": ",".join(top3_models),
                })

        # inv-MAE weighted
        if all(m in g.columns for m in top3_models):
            w = 1.0 / top3_mae.loc[top3_models].values
            w = w / w.sum()
            inv_pred = (g[top3_models].values * w).sum(axis=1)
            for i, row in g.reset_index().iterrows():
                rows_top3_invmae.append({
                    "disease": d, "horizon": h, "origin": row["origin"],
                    "cd_mun": row["cd_mun"], "time_idx": row["time_idx"],
                    "y_true": row["y_true"], "y_pred": inv_pred[i],
                    "members": ",".join(f"{m}:{wi:.2f}" for m, wi in zip(top3_models, w)),
                })

        # all-models mean (sanity)
        cols = [c for c in all_models if c in g.columns]
        if cols:
            am = g[cols].mean(axis=1)
            for i, row in g.reset_index().iterrows():
                rows_all_mean.append({
                    "disease": d, "horizon": h, "origin": row["origin"],
                    "cd_mun": row["cd_mun"], "time_idx": row["time_idx"],
                    "y_true": row["y_true"], "y_pred": am.iloc[i],
                    "members": "all",
                })

    return {
        "top3_mean": pd.DataFrame(rows_top3_mean),
        "top3_inv_mae": pd.DataFrame(rows_top3_invmae),
        "all_mean": pd.DataFrame(rows_all_mean),
    }

ensembles = build_ensembles(preds, val_rank)
for name, df in ensembles.items():
    df.to_csv(ENS_DIR / "tables" / f"ensemble_{name}_predictions.csv", index=False)
    print(f"{name}: {len(df):,} previsões")

# %% [markdown]
# ## 4. Avaliação no conjunto de teste (2021, 2022)

# %%
def score_predictions_long(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for (d, h), g in df.groupby(["disease", "horizon"]):
        ev = evaluate(g["y_true"].values, g["y_pred"].values, name=name, disease=d, horizon=h)
        rows.append({"model": name, "disease": d, "horizon": h, "mae": ev["mae"], "rmse": ev["rmse"], "n": ev["n"]})
    return pd.DataFrame(rows)

# scores no teste: ensembles + cada modelo individual
test_preds = preds[preds["origin"].isin(TEST_ORIGINS)]
single_scores = []
for m in test_preds["model"].unique():
    g = test_preds[test_preds["model"] == m]
    single_scores.append(score_predictions_long(g, m))
single_scores = pd.concat(single_scores, ignore_index=True)

ens_scores = pd.concat(
    [score_predictions_long(df, name) for name, df in ensembles.items()],
    ignore_index=True,
)

all_scores = pd.concat([single_scores, ens_scores], ignore_index=True)
all_scores.to_csv(ENS_DIR / "tables" / "test_scores.csv", index=False)

print("\nMAE médio no TESTE (2021-2022) por horizonte:")
print(all_scores.pivot_table(index="model", columns="horizon", values="mae").round(2).sort_values(1))

# %% [markdown]
# ## 5. O ensemble bate o melhor modelo individual?

# %%
best_single = (single_scores
               .sort_values("mae")
               .groupby(["disease", "horizon"]).head(1)
               .rename(columns={"mae": "mae_best_single", "model": "best_model"})
               [["disease", "horizon", "best_model", "mae_best_single"]])

ens_top3 = ens_scores[ens_scores["model"] == "top3_inv_mae"].rename(columns={"mae": "mae_ensemble"})[["disease", "horizon", "mae_ensemble"]]
cmp = best_single.merge(ens_top3, on=["disease", "horizon"])
cmp["delta_mae"] = cmp["mae_ensemble"] - cmp["mae_best_single"]
cmp["ensemble_wins"] = cmp["delta_mae"] < 0
cmp.to_csv(ENS_DIR / "tables" / "ensemble_vs_best_single.csv", index=False)
print(cmp.round(3).to_string(index=False))
print(f"\nEnsemble (top3_inv_mae) vence em {cmp['ensemble_wins'].sum()}/{len(cmp)} pares (doença × horizonte).")

# %% [markdown]
# ## 6. Visualização: ensemble vs melhor single, por horizonte

# %%
fig, axes = plt.subplots(1, len(DISEASES), figsize=(22, 4.5), sharey=False)
for ax, d in zip(axes, DISEASES):
    sub = all_scores[all_scores["disease"] == d]
    pv = sub.pivot_table(index="horizon", columns="model", values="mae")
    # destacar ensembles
    for col in pv.columns:
        lw = 2.2 if col.startswith(("top3_", "all_")) else 0.9
        alpha = 1.0 if col.startswith(("top3_", "all_")) else 0.55
        pv[col].plot(ax=ax, marker="o", lw=lw, alpha=alpha, label=col)
    ax.set_title(d); ax.set_xlabel("horizonte"); ax.set_ylabel("MAE")
    ax.legend(fontsize=6, ncol=2)
fig.suptitle("MAE no teste (2021-2022) — ensemble vs single (linhas grossas = ensembles)", y=1.02)
fig.tight_layout()
fig.savefig(ENS_DIR / "figures" / "ensemble_vs_single_mae.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Forecast plot do ensemble por município (top-3 inv-MAE)

# %%
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-12-01")

def plot_ensemble_forecasts(panel, ens_df, disease):
    col = f"n_{disease}"
    sub = ens_df[ens_df["disease"] == disease].copy()
    sub["date"] = sub["time_idx"].apply(lambda i: pd.Timestamp("2000-01-01") + pd.DateOffset(months=int(i)))
    sub["cd_mun"] = sub["cd_mun"].astype(str).str.zfill(7)

    munis = sorted(panel["nm_mun"].dropna().unique())
    fig, axes = plt.subplots(6, 4, figsize=(22, 22), sharex=False)
    for ax, muni in zip(axes.flat, munis):
        cd = next((k for k, v in MUNICIPIOS_ALVO.items() if v == muni), None)
        if not cd:
            ax.axis("off"); continue
        hist = panel[panel["cd_mun"] == cd].sort_values("date")
        ax.plot(hist["date"], hist[col], lw=0.8, color="#444", label="observado")
        pm = sub[sub["cd_mun"] == cd]
        for i, (origin, g) in enumerate(pm.groupby("origin")):
            g = g.sort_values("time_idx")
            ax.plot(g["date"], g["y_pred"], marker=".", ms=4, lw=1.4,
                    color=plt.cm.tab10.colors[i % 10], label=origin)
        ax.axvspan(COVID_START, COVID_END, color="red", alpha=0.05)
        ax.set_title(muni, fontsize=9); ax.tick_params(labelsize=7)
    for ax in axes.flat[len(munis):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="upper left", ncol=2)
    fig.suptitle(f"{disease} — ensemble top3_inv_mae vs observado (origens em cores)", y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(ENS_DIR / "figures" / f"ensemble_forecast_{disease}.png", dpi=110, bbox_inches="tight")
    plt.show()

panel = pd.read_parquet(PROCESSED / "panel_23munis.parquet")
panel["cd_mun"] = panel["cd_mun"].astype(str).str.zfill(7)

ens_top3_inv = ensembles["top3_inv_mae"]
for d in DISEASES:
    plot_ensemble_forecasts(panel, ens_top3_inv, d)

# %% [markdown]
# ## 8. Tabela por município (MAE do ensemble vs melhor single)

# %%
def per_muni_mae(df, name):
    rows = []
    for (d, cd), g in df.groupby(["disease", "cd_mun"]):
        ev = evaluate(g["y_true"].values, g["y_pred"].values, name=name, disease=d)
        rows.append({"model": name, "disease": d, "cd_mun": cd,
                     "nm_mun": MUNICIPIOS_ALVO.get(cd, cd), "mae": ev["mae"]})
    return pd.DataFrame(rows)

ens_per_muni = per_muni_mae(ensembles["top3_inv_mae"], "top3_inv_mae")
single_per_muni_rows = []
for m in test_preds["model"].unique():
    g = test_preds[test_preds["model"] == m]
    single_per_muni_rows.append(per_muni_mae(g, m))
single_per_muni = pd.concat(single_per_muni_rows, ignore_index=True)

best_single_per_muni = (single_per_muni
                        .sort_values("mae")
                        .groupby(["disease", "cd_mun"]).head(1)
                        .rename(columns={"model": "best_single", "mae": "mae_best_single"})
                        [["disease", "cd_mun", "nm_mun", "best_single", "mae_best_single"]])
muni_cmp = best_single_per_muni.merge(
    ens_per_muni.rename(columns={"mae": "mae_ensemble"})[["disease", "cd_mun", "mae_ensemble"]],
    on=["disease", "cd_mun"],
)
muni_cmp["delta"] = muni_cmp["mae_ensemble"] - muni_cmp["mae_best_single"]
muni_cmp.to_csv(ENS_DIR / "tables" / "per_muni_ensemble_vs_best.csv", index=False)
print("\nPor município (top 10 menores MAE de ensemble):")
print(muni_cmp.sort_values("mae_ensemble").head(10).round(3).to_string(index=False))

# %% [markdown]
# ## 9. Síntese final

# %%
print("\n=== RESUMO ENSEMBLE ===")
print(f"\nMAE médio no teste (2021-2022) por estratégia:")
ens_only = all_scores[all_scores["model"].isin(["top3_mean", "top3_inv_mae", "all_mean"])]
print(ens_only.groupby("model")["mae"].mean().round(3).to_string())

print(f"\nMAE médio do melhor modelo individual no teste: {single_scores['mae'].groupby(single_scores['model']).mean().min():.3f}")
print(f"\nVitórias do ensemble top3_inv_mae sobre o melhor single: {cmp['ensemble_wins'].sum()}/{len(cmp)}")

# salva síntese curta
with open(ENS_DIR / "README.md", "w", encoding="utf-8") as fp:
    fp.write("# Ensemble — top-3 por (doença × horizonte)\n\n")
    fp.write("**Split:** validação 2019-2020 (escolhe top-3 + pesos), teste 2021-2022 (avalia).\n\n")
    fp.write("## Estratégias\n\n")
    fp.write("- `top3_mean` — média simples dos 3 melhores na validação\n")
    fp.write("- `top3_inv_mae` — média ponderada por 1/MAE\n")
    fp.write("- `all_mean` — sanidade: média de todos os modelos\n\n")
    fp.write("## MAE médio no teste\n\n")
    fp.write(ens_only.groupby("model")["mae"].mean().round(3).to_markdown())
    fp.write("\n\n## Ensemble vs melhor single\n\n")
    fp.write(cmp.round(3).to_markdown(index=False))
    fp.write(f"\n\nEnsemble vence em **{cmp['ensemble_wins'].sum()}/{len(cmp)}** pares.\n")

print(f"\nresultados em {ENS_DIR}")
