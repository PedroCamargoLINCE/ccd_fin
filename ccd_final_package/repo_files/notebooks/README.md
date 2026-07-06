# Notebooks

## `train_all.ipynb`

Pipeline completo: EDA → baselines (7: SeasonalNaive, SeasonalMA3, SARIMA, LightGBM,
CatBoost, XGBoost, Prophet) → deep panel (5: TFT, N-HiTS, DeepAR, LSTM, GRU) →
comparação → diagnóstico por município × doença.

> Para rodar o pipeline inteiro sem abrir o notebook, use `bash run_experiments.sh` na
> raiz do repo — é o entry point recomendado para reprodução (instala deps faltantes,
> monta o painel, roda tudo, gera `results/`).

**Como rodar via notebook:**
1. Abra `train_all.ipynb` no VSCode (ou gere a partir do `.py`, ver abaixo).
2. Selecione o kernel **Python (ccd)**.
3. `Run All` (ou execute célula a célula).

**Cache:** cada bloco checa se os artefatos já existem em `reports/` e pula o trabalho.
Para forçar re-treino de um modelo, apague o `.csv` correspondente em
`reports/deep_<model>_<disease>.csv`.

**Saída:**
- `reports/figures/` — PNGs (heatmaps, séries, correlações, previsões, diagnóstico por origem).
- `reports/baselines.csv`, `reports/baselines_long.csv` — métricas + predições point-level dos baselines.
- `reports/deep_<model>_<disease>.csv` — predições point-level dos deep models.
- Consolidação final: rode `python scripts/make_results.py` — gera
  `results/tables/final_summary.csv` (com recall, DTW, desvio-padrão) e as figuras
  comparativas.

**Parâmetros que vale tunar na célula "Deep panel — TFT e N-HiTS":**
- `DEEP_EPOCHS = 200` → teto; `EarlyStopping` (patience=7 nessa célula) corta bem antes na prática.
- `DEEP_MODELS = ["tft", "nhits", "deepar", "lstm", "gru"]` → adicione/remova modelos aqui.
- `DEEP_BATCH = 128`, `DEEP_HIDDEN = 32` → suba `DEEP_HIDDEN` para 64-128 se tiver VRAM sobrando.

Falhas de treino de um modelo específico (ex.: incompatibilidade de biblioteca) são
capturadas e logadas com `[SKIP] <modelo>/<doença>: ...` — não derrubam o run inteiro.

## `ensemble.py`

Top-3 por (doença × horizonte), com split honesto (validação 2019-2020, teste
2021-2022). Lê de `reports/*.csv` — portanto precisa que `train_all` já tenha rodado
(ou pelo menos os modelos que você quer incluir no ensemble). **Precisa ser re-rodado**
sempre que o conjunto de modelos em `reports/` mudar (ex.: depois de adicionar um modelo
novo ou consertar um que estava quebrado) — os resultados em `results/ensemble/` não se
atualizam sozinhos.

## `build_notebook.py`

Converte `train_all.py` (fonte editável com `# %%` cells) → `train_all.ipynb`. Se quiser
editar o notebook, edite o `.py` e rode:

```
python notebooks/build_notebook.py
```
