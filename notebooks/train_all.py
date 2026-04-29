"""
Notebook orquestrador: treina todos os modelos, avalia e gera diagnósticos
por município × doença.

Este .py usa markers # %% — abra no VSCode como notebook (Python: Convert
to Jupyter Notebook) ou rode `python notebooks/build_notebook.py` para
gerar `train_all.ipynb`.

Reexecução: carrega cache (parquet + csvs de reports) se existir; recomputa
só o que estiver faltando.
"""
# %% [markdown]
# # Predição multi-doença para 23 municípios de SP
#
# Pipeline completo: EDA → baselines (naive, SARIMA, LightGBM) → deep
# panel (TFT, N-HiTS) → comparação → diagnósticos por município.
#
# **Alvo:** contagens mensais (NB-compatível) de 5 doenças
# (hanseníase, hepatite, HIV/AIDS, sífilis, tuberculose).
#
# **Decisões-chave:**
# - Treinar em contagens (n_*) com normalizador por município; ajustar
#   taxas derivando n/populacao * 100k na visualização final.
# - Rolling origin com 4 origens (2019-12, 2020-12, 2021-12, 2022-12).
# - Horizontes: 1, 3, 6, 12 meses.

# %%
import os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# garantir que src/ é importável quando rodar do notebooks/
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
from src.utils.splits import DEFAULT_ORIGINS, DEFAULT_HORIZONS, rolling_origin

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"reports    : {REPORTS}")
print(f"figures    : {FIGURES}")

# %% [markdown]
# ## 1. Carregar ou construir painel
# Usa cache em parquet; reconstrói a partir dos Excel se não existir.

# %%
panel_path = PROCESSED / "panel_23munis.parquet"
if not panel_path.exists():
    from src.data.load import build_cache
    from src.data.build_panel import build_panel
    build_cache()
    build_panel()

panel = pd.read_parquet(panel_path)
print(f"shape: {panel.shape}")
print(f"municípios: {panel['cd_mun'].nunique()}")
print(f"período: {panel['date'].min():%Y-%m} → {panel['date'].max():%Y-%m}")
panel.head()

# %% [markdown]
# ## 2. EDA — fração de zeros
# Decide a estratégia de loss por doença.

# %%
def zero_fraction_table(panel):
    rows = []
    for d in DISEASES:
        for cd, g in panel.groupby("cd_mun"):
            v = g[f"n_{d}"].dropna()
            rows.append({
                "doenca": d, "cd_mun": cd, "nm_mun": g["nm_mun"].iloc[0],
                "zero_frac": (v == 0).mean(),
                "mean": v.mean(), "max": v.max(),
                "var_mean_ratio": v.var() / max(v.mean(), 1e-9),
            })
    return pd.DataFrame(rows)

zf = zero_fraction_table(panel)
summary = zf.groupby("doenca").agg(
    zero_median=("zero_frac", "median"),
    zero_max=("zero_frac", "max"),
    varmean_med=("var_mean_ratio", "median"),
    mean_count=("mean", "mean"),
).round(3)
print("Resumo por doença:")
print(summary)

fig, ax = plt.subplots(figsize=(9, 8))
piv = zf.pivot(index="nm_mun", columns="doenca", values="zero_frac")
sns.heatmap(piv, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1,
            cbar_kws={"label": "fração de meses com 0 casos"}, ax=ax)
ax.set_title("Zero-fraction por município × doença")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(FIGURES / "zero_fraction_heatmap.png", dpi=140)
plt.show()

# %% [markdown]
# ## 3. EDA — séries temporais por município
# Uma grade 6×4 por doença (23 munis + um painel vazio).

# %%
COVID_START = pd.Timestamp("2020-03-01")
COVID_END   = pd.Timestamp("2021-12-01")

def plot_disease_series(panel, disease, use_rate=True):
    col = f"tx_{disease}" if use_rate else f"n_{disease}"
    fig, axes = plt.subplots(6, 4, figsize=(22, 20), sharex=True)
    munis = sorted(panel["nm_mun"].dropna().unique())
    for ax, muni in zip(axes.flat, munis):
        g = panel[panel["nm_mun"] == muni].sort_values("date")
        ax.plot(g["date"], g[col], lw=0.8, color="#2a6df4")
        ax.axvspan(COVID_START, COVID_END, color="red", alpha=0.08)
        ax.set_title(muni, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(munis):]:
        ax.axis("off")
    fig.suptitle(f"{'Taxa (por 100k)' if use_rate else 'Contagem'} — {disease}  (faixa vermelha = COVID)", y=1.00, fontsize=13)
    fig.tight_layout()
    out = FIGURES / f"series_{disease}_{'tx' if use_rate else 'n'}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"saved {out.name}")

for d in DISEASES:
    plot_disease_series(panel, d, use_rate=True)

# %% [markdown]
# ## 4. Correlação clima × alvo (lag 0 a 6)

# %%
def lag_correlation(panel, disease, max_lag=6):
    vars_ = ["evapot", "precip", "temp_min", "temp_max", "umid"]
    col = f"tx_{disease}"
    rows = []
    for cv in vars_:
        for lag in range(max_lag + 1):
            corrs = []
            for _, g in panel.groupby("cd_mun"):
                g = g.sort_values("date")
                c = g[col].corr(g[cv].shift(lag), method="spearman")
                if pd.notna(c): corrs.append(c)
            rows.append({"clima": cv, "lag": lag, "rho": np.mean(corrs) if corrs else np.nan})
    return pd.DataFrame(rows)

corr_all = pd.concat([lag_correlation(panel, d).assign(doenca=d) for d in DISEASES])

fig, axes = plt.subplots(1, len(DISEASES), figsize=(22, 4.2), sharey=True)
for ax, d in zip(axes, DISEASES):
    sub = corr_all[corr_all["doenca"] == d].pivot(index="clima", columns="lag", values="rho")
    sns.heatmap(sub, ax=ax, cmap="RdBu_r", center=0, vmin=-0.1, vmax=0.1, annot=True, fmt=".02f", cbar=False)
    ax.set_title(d)
fig.suptitle("Spearman(clima lagado, taxa) — média entre municípios", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "climate_lag_corr.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Rolling origin — visualização

# %%
fig, ax = plt.subplots(figsize=(13, 2.5))
ax.axvline(panel["date"].min(), color="gray", alpha=0.3)
ax.axvline(panel["date"].max(), color="gray", alpha=0.3)
colors = plt.cm.tab10.colors
for i, split in enumerate(rolling_origin()):
    h = max(split.horizons)
    test_end = split.train_end + pd.DateOffset(months=h)
    ax.axvspan(panel["date"].min(), split.train_end, ymin=0.08 + i*0.2, ymax=0.08 + i*0.2 + 0.16, alpha=0.2, color=colors[i])
    ax.axvspan(split.train_end, test_end, ymin=0.08 + i*0.2, ymax=0.08 + i*0.2 + 0.16, alpha=0.55, color=colors[i])
    ax.text(panel["date"].min(), 0.08 + i*0.2 + 0.08, f" {split.name}", va="center", fontsize=9)
ax.set_yticks([])
ax.set_title("Rolling origin — treino (claro) × teste (escuro)")
fig.tight_layout()
fig.savefig(FIGURES / "rolling_origin.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Baselines — naive sazonal + SARIMA + LightGBM

# %%
BASELINE_CSV = REPORTS / "baselines.csv"
if not BASELINE_CSV.exists():
    from src.models.baselines import run_all_baselines
    baselines = run_all_baselines(panel)
    baselines.to_csv(BASELINE_CSV, index=False)
else:
    baselines = pd.read_csv(BASELINE_CSV)
    print(f"loaded cached baselines from {BASELINE_CSV.name}")

print("\nMAE médio por (modelo, horizonte):")
pv_mae = baselines.pivot_table(index="model", columns="horizon", values="mae", aggfunc="mean").round(2)
print(pv_mae)

fig, ax = plt.subplots(figsize=(8, 4))
pv_mae.T.plot(ax=ax, marker="o")
ax.set_xlabel("horizonte (meses)")
ax.set_ylabel("MAE (contagem)")
ax.set_title("Baselines — MAE por horizonte (média entre doenças × origens)")
ax.legend(title="modelo")
fig.tight_layout()
fig.savefig(FIGURES / "baselines_mae_by_horizon.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Deep panel — TFT e N-HiTS
#
# Treino por doença × origem. Configurar `max_epochs` menor para re-runs
# rápidos; maior para resultados finais.

# %%
DEEP_EPOCHS = 200           # teto — EarlyStopping (patience=15) corta antes
DEEP_BATCH  = 128
DEEP_HIDDEN = 32
DEEP_MODELS = ["tft", "nhits", "deepar"]

def run_deep_all(panel, models=DEEP_MODELS, diseases=None, epochs=DEEP_EPOCHS):
    diseases = diseases or DISEASES
    all_out = []
    if models:
        from src.models.deep_panel import run_deep_single
    for mdl in models:
        for d in diseases:
            out_csv = REPORTS / f"deep_{mdl}_{d}.csv"
            if out_csv.exists():
                df = pd.read_csv(out_csv)
                print(f"[cached] {out_csv.name} ({len(df)} rows)")
            else:
                print(f"\n===== {mdl.upper()} / {d} =====")
                df = run_deep_single(panel, disease=d, model_name=mdl,
                                     cfg_overrides={"max_epochs": epochs, "patience": 7,
                                                    "batch_size": DEEP_BATCH, "hidden_size": DEEP_HIDDEN})
                df.to_csv(out_csv, index=False)
            df["model"] = mdl
            df["disease"] = d
            all_out.append(df)
    if not all_out:
        print("(nenhum modelo deep foi rodado — DEEP_MODELS vazio)")
        return pd.DataFrame(columns=["model","disease","horizon","origin","cd_mun","time_idx","y_true","y_pred"])
    return pd.concat(all_out, ignore_index=True)

deep_df = run_deep_all(panel)
print(f"\ndeep results: {deep_df.shape}")

# %% [markdown]
# ## 8. Comparação final: baselines vs deep

# %%
from src.eval.metrics import evaluate

def summarize(deep_df, baselines_df):
    rows = []
    # baselines já vem agregado por modelo/origem/doença/horizonte
    for (mdl, d, h), g in baselines_df.groupby(["model", "disease", "horizon"]):
        rows.append({"model": mdl, "disease": d, "horizon": h,
                     "mae": g["mae"].mean(), "rmse": g["rmse"].mean(), "smape": g["smape"].mean()})
    # deep é long (y_true, y_pred) — agrega por (model × disease × horizon)
    for (mdl, d, h), g in deep_df.groupby(["model", "disease", "horizon"]):
        m = evaluate(g["y_true"].values, g["y_pred"].values, name=mdl, disease=d, horizon=h)
        rows.append({"model": mdl, "disease": d, "horizon": h,
                     "mae": m["mae"], "rmse": m["rmse"], "smape": m["smape"]})
    return pd.DataFrame(rows)

summary = summarize(deep_df, baselines)
summary.to_csv(REPORTS / "final_summary.csv", index=False)
print("Top 3 modelos por doença × horizonte (menor MAE):")
top = (summary.sort_values("mae")
              .groupby(["disease", "horizon"])
              .head(3))
for (d, h), g in top.groupby(["disease", "horizon"]):
    print(f"  {d} h={h:02d}: " + " | ".join(f"{row['model']}={row['mae']:.2f}" for _, row in g.iterrows()))

# %% [markdown]
# ### 8.1 Tabela wide: MAE por modelo × (doença, horizonte)

# %%
pv = summary.pivot_table(index="model", columns=["disease", "horizon"], values="mae").round(2)
print(pv)

fig, axes = plt.subplots(1, len(DISEASES), figsize=(22, 5), sharey=False)
for ax, d in zip(axes, DISEASES):
    sub = summary[summary["disease"] == d]
    piv = sub.pivot_table(index="horizon", columns="model", values="mae")
    piv.plot(ax=ax, marker="o")
    ax.set_title(d)
    ax.set_xlabel("horizonte (meses)")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=7, title=None)
fig.suptitle("MAE por horizonte — comparação entre modelos", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES / "comparison_mae_by_horizon.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Visualização por município — predição vs real (melhor modelo)

# %%
def best_model_per_disease(summary):
    agg = summary.groupby(["disease", "model"])["mae"].mean().reset_index()
    return {d: g.sort_values("mae").iloc[0]["model"] for d, g in agg.groupby("disease")}

best_map = best_model_per_disease(summary)
print("Melhor modelo por doença (agregando horizontes):")
for d, m in best_map.items():
    print(f"  {d}: {m}")

def plot_predictions_grid(panel, deep_df, disease, model_name):
    col = f"n_{disease}"
    sub = deep_df[(deep_df["disease"] == disease) & (deep_df["model"] == model_name)].copy()
    if sub.empty:
        print(f"  (sem previsões deep para {disease}/{model_name} — pulando)")
        return
    sub["date"] = pd.to_datetime(panel["date"].min()) + pd.to_timedelta(sub["time_idx"] * 30.4375, unit="D")
    # arredondar para o primeiro dia do mês correspondente
    sub["date"] = sub["time_idx"].apply(lambda i: pd.Timestamp("2000-01-01") + pd.DateOffset(months=int(i)))

    munis = sorted(panel["nm_mun"].dropna().unique())
    fig, axes = plt.subplots(6, 4, figsize=(22, 22), sharex=False)
    for ax, muni in zip(axes.flat, munis):
        cd = [k for k, v in MUNICIPIOS_ALVO.items() if v == muni][0]
        hist = panel[panel["cd_mun"] == cd].sort_values("date")
        ax.plot(hist["date"], hist[col], lw=0.8, color="#666666", label="observado")
        pm = sub[sub["cd_mun"] == cd]
        # uma cor por origem
        for i, (origin, g) in enumerate(pm.groupby("origin")):
            g = g.sort_values("time_idx")
            ax.plot(g["date"], g["y_pred"], marker=".", ms=4, lw=1.2, color=plt.cm.tab10.colors[i % 10], label=origin)
        ax.axvspan(COVID_START, COVID_END, color="red", alpha=0.05)
        ax.set_title(muni, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(munis):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="upper left", ncol=2)
    fig.suptitle(f"{disease} — {model_name} vs observado (origens em cores)", y=1.00, fontsize=13)
    fig.tight_layout()
    out = FIGURES / f"forecast_{disease}_{model_name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.show()
    print(f"  saved {out.name}")

for d, m in best_map.items():
    if m in ("tft", "nhits"):
        plot_predictions_grid(panel, deep_df, d, m)

# %% [markdown]
# ## 10. Diagnósticos — erro por origem (choque COVID)

# %%
def error_by_origin(deep_df, baselines):
    bl = baselines.groupby(["model", "disease", "horizon", "origin"])["mae"].mean().reset_index()
    dp_rows = []
    for (mdl, d, h, o), g in deep_df.groupby(["model", "disease", "horizon", "origin"]):
        m = evaluate(g["y_true"].values, g["y_pred"].values, name=mdl, disease=d, horizon=h)
        dp_rows.append({"model": mdl, "disease": d, "horizon": h, "origin": o, "mae": m["mae"]})
    return pd.concat([bl, pd.DataFrame(dp_rows)], ignore_index=True)

by_origin = error_by_origin(deep_df, baselines)
by_origin["origin_year"] = by_origin["origin"].str.extract(r"(\d{4})")[0].astype(int)

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=by_origin.query("horizon == 3"), x="origin_year", y="mae",
             hue="model", marker="o", ax=ax)
ax.set_title("MAE (h=3) por origem — estabilidade temporal dos modelos")
ax.set_xlabel("ano da origem")
fig.tight_layout()
fig.savefig(FIGURES / "stability_by_origin.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 11. Conclusões e próximos passos
#
# - Tabela final em `reports/final_summary.csv`.
# - Figuras em `reports/figures/`.
# - Para incrementar:
#   - **Mais epochs** nos deep models (aqui `DEEP_EPOCHS = 25`; testar 80+ em produção).
#   - **Hyperparameter search** com Optuna em hidden_size, dropout, lr, encoder_length.
#   - **ST-GNN** como ablação (23 nós é pequeno, espera-se ganho modesto — vale só se TFT for insuficiente).
#   - **Post-hoc calibration** via conformal prediction para intervalos calibrados.
#   - **Ensemble** do top-3 por doença/horizonte.

print("pipeline completo — todos os artefatos em reports/")
