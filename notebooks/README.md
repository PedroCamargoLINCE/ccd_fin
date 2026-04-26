# Notebooks

## `train_all.ipynb`

Pipeline completo: EDA → baselines → TFT/N-HiTS → comparação → diagnóstico por município × doença.

**Como rodar:**
1. Abra `train_all.ipynb` no VSCode.
2. Selecione o kernel **Python (ccd)**.
3. `Run All` (ou execute célula a célula).

**Cache:** cada bloco checa se os artefatos já existem em `reports/` e pula o trabalho. Para forçar re-treino de um modelo, apague o `.csv` correspondente em `reports/deep_<model>_<disease>.csv`.

**Saída:**
- `reports/figures/` — PNGs (heatmaps, séries, correlações, previsões, diagnóstico por origem).
- `reports/baselines.csv` — métricas dos baselines.
- `reports/deep_<model>_<disease>.csv` — predições point-level.
- `reports/final_summary.csv` — tabela consolidada modelo × doença × horizonte.

**Parâmetros que vale tunar na célula 7:**
- `DEEP_EPOCHS = 25` → suba para 60-80 para resultados finais (demora ~5 min/doença por origem na sua GPU).
- `DEEP_MODELS = ["tft", "nhits"]` → adicione/remova modelos.
- `DEEP_HIDDEN = 32` → 64-128 em produção (há VRAM sobrando com 24 GB × 2).

## `build_notebook.py`

Converte `train_all.py` (fonte editável com `# %%` cells) → `train_all.ipynb`. Se quiser editar o notebook, edite o `.py` e rode:

```
python notebooks/build_notebook.py
```
