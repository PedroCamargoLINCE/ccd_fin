# ccd_fin — predição mensal de doenças em 23 municípios de SP

Pipeline de previsão temporal multi-doença (hanseníase, hepatite, HIV/AIDS, sífilis,
tuberculose) em 23 municípios do estado de São Paulo, com features climáticas e
socioeconômicas. Implementado em PyTorch com `pytorch-forecasting` + Lightning,
comparado contra 7 baselines clássicos e de árvore.

---

## 1. Objetivo

Predizer a **incidência mensal** das 5 doenças nos 23 municípios-alvo (Barueri, Bauru,
Campinas, Carapicuíba, Diadema, Guarujá, Guarulhos, Itapevi, Jundiaí, Mauá, Osasco,
Paulínia, Praia Grande, Ribeirão Preto, Santo André, Santos, São Bernardo do Campo, São
José do Rio Preto, São José dos Campos, São Paulo, São Vicente, Sorocaba, Taboão da
Serra) em horizontes de **1, 3, 6 e 12 meses**.

## 2. Dados

| Grupo | Granularidade | Período | Fonte |
|---|---|---|---|
| **Doenças (contagens + taxas)** | mensal | 2000-01 → 2023-12 (288 meses) | `<doenca>_00_23.xlsx` + `Taxas/TX_*.xlsx` |
| **Clima** (evapot, precip, T_min, T_max, umid) | mensal | 1999-01 → 2023-12 | 5 arquivos `Evapot_SP.xlsx`, `Precip_SP.xlsx`, etc. |
| **Socioeconômico** (pop, dens. demog., PPC, urbanização) | anual | 2000 → 2023 | `Pop_Geral_SP.xlsx`, `Dens_demog_SP.csv`, etc. |

**645 municípios de SP** em todos os arquivos (recortamos os 23 alvos).

**Painel final:** 23 munis × 288 meses = **6.624 linhas × 26 colunas**, 0% missing nos alvos.

## 3. Descobertas da EDA

### 3.1 Zero-inflation severa em algumas doenças

Fração de meses com 0 casos reportados (mediana entre os 23 munis):

| Doença | Zero-fraction (mediana / máx) | Var/Mean |
|---|---|---|
| **hanseníase** | **88% / 97%** | 1.2 |
| hepatite | 70% / 93% | 2.4 |
| sífilis | 41% / 88% | 2.7 |
| tuberculose | 14% / 41% | 3.0 |
| HIV/aids | 8% / 61% | 3.8 |

Sobredispersão (var/média ≫ 1) em todas as doenças → Poisson é insuficiente; os
modelos de árvore usam objetivo Poisson/Tweedie e o DeepAR usa NegativeBinomial nativo.

### 3.2 Clima tem correlação fraca com incidência

Spearman médio entre `clima(t-k)` e `taxa(t)` para `k ∈ [0..6]`: **|ρ| < 0.08** na
maioria dos pares — ver `results/figures/eda/climate_lag_corr.png`. Umidade é o sinal
mais consistente com hepatite e HIV. Clima é incluído nos modelos (defasado, sem
vazamento — ver §4), mas não se espera ganho grande dele isolado.

### 3.3 Choque COVID (2020-03 → 2021-12)

Reporting de doenças cai abruptamente nesse período. Tratado como flag
(`covid_period`), incluído como covariável conhecida no futuro (é determinística).

## 4. Design experimental e validação

- **Alvo:** contagens mensais (`n_<doenca>`); taxas derivadas para visualização.
- **Split temporal:** **rolling origin** com 4 origens (`2019-12`, `2020-12`, `2021-12`,
  `2022-12`). Treina até a origem, testa nos próximos 12 meses.
- **Métricas:** MAE, RMSE, R², SMAPE, recall/precisão de eventos não-zero (detecção em
  série zero-inflada), DTW (aderência de forma da curva) — agregadas por
  (modelo × doença × horizonte), com desvio-padrão entre origens e (em runs multi-seed)
  entre sementes.

### Vazamentos corrigidos (revisão de validação, 2026-07)

Uma auditoria encontrou e corrigiu 4 vazamentos que inflavam artificialmente os deep
models e viesavam a comparação com os baselines de árvore:

1. **Janela de teste sempre a mesma.** Os deep models antes previam sempre os últimos
   12 meses do painel, independente da origem — não era rolling-origin de verdade.
   Corrigido: o decoder de teste agora cai exatamente em `[T+1, T+H]` de cada origem.
2. **Vazamento de clima futuro.** Clima entrava como covariável "conhecida no futuro" —
   o modelo via o clima real da janela que estava prevendo. Corrigido: clima agora é
   só encoder (passado), exceto para o DeepAR (ver nota abaixo).
3. **Teste usado no early stopping.** O mesmo dataset de teste decidia quando parar o
   treino. Corrigido: validação de early stopping usa uma janela separada
   `[T-H+1, T]`, que nunca toca o teste.
4. **Vazamento de lag nos modelos de árvore para h>1.** As features de lag eram
   calculadas na data-alvo (T+h), usando valores posteriores à origem. Corrigido:
   esquema "as-of-origem" — features sempre relativas à origem T, um modelo por
   horizonte.

> **Nota técnica — DeepAR.** O DeepAR decodifica autorregressivamente e não aceita
> covariável "desconhecida no futuro" além do próprio alvo (`AssertionError: Encoder
> and decoder variables have to be the same apart from target variable`). Por isso ele
> roda com alvo + calendário apenas (sem clima), o mesmo esquema usado no LSTM/GRU.

## 5. Modelos implementados

> **Detalhes completos de arquitetura e hiperparâmetros** (janelas de entrada, horizonte,
> batch size, learning rate, épocas, topologia exata de cada modelo, defaults herdados da
> biblioteca e tuning) em **[`docs/ARQUITETURAS.md`](docs/ARQUITETURAS.md)** — é a referência
> para a seção de métodos do artigo.

### Baselines (`src/models/baselines.py`)
- **SeasonalNaive**, **SeasonalMA3** — âncoras de sanidade
- **AutoARIMA/SARIMA** (`statsforecast`, sazonalidade 12) — cobre ARIMA (caso especial)
- **LightGBM**, **CatBoost**, **XGBoost** — Poisson, features de lag as-of-origem
- **Prophet** — univariado por município, sazonalidade anual

### Deep panel (`src/models/deep_panel.py`)
- **TFT** (`TemporalFusionTransformer`), **N-HiTS** — `QuantileLoss`, clima como covariável de encoder
- **DeepAR** — `NegativeBinomialDistributionLoss`, alvo + calendário (sem clima, ver nota acima)
- **LSTM**, **GRU** (via `RecurrentNetwork`) — alvo + calendário

Todos com **EarlyStopping** (patience) e **ReduceLROnPlateau**. `max_epochs=200` é teto;
convergência típica bem antes disso.

### Ensemble (`notebooks/ensemble.py`)
Top-3 por (doença × horizonte), validação 2019-2020 / teste 2021-2022. **Precisa ser
re-rodado** sempre que o conjunto de modelos mudar — ver `results/ensemble/README.md`.

### Não implementado (roadmap)
- **ZINB explícito** para hanseníase/hepatite/sífilis (zero-inflação extrema).
- **Importância de covariáveis nos deep models** — o hook do TFT (`interpret_output`)
  já está no código; falta rodar em GPU para gerar os números. Para as árvores já está
  feito: ver `results/tables/covariate_importance/`.
- **ST-GNN**: com 23 nós, ganho marginal vs. custo de integração — descartado.

## 6. Resultados

**A tabela completa e sempre atualizada vive em [`results/README.md`](results/README.md)**
e nas tabelas em [`results/tables/`](results/tables/) (`final_summary.csv`,
`dtw_by_model.csv`, `per_muni_*.csv`) — geradas automaticamente por
`scripts/make_results.py`. Não duplicamos números aqui para evitar desatualização;
consulte o link acima para o MAE/RMSE/R²/recall/DTW mais recente por modelo × doença ×
horizonte, e o vencedor por doença.

Figuras em [`results/figures/`](results/figures/) (EDA, forecasts, comparação).

## 7. Estrutura do repo

```
src/
├── data/
│   ├── load.py          # leitores wide→long (xlsx strict OOXML, ods, csv)
│   ├── build_panel.py   # painel unificado 23 munis × 288 meses
│   └── eda.py           # zero-fraction, missingness, correlação clima×alvo
├── models/
│   ├── baselines.py     # SeasonalNaive/MA3, SARIMA, LightGBM, CatBoost, XGBoost, Prophet
│   └── deep_panel.py    # TFT, N-HiTS, DeepAR, LSTM, GRU via pytorch-forecasting
├── eval/
│   └── metrics.py       # MAE, RMSE, R², SMAPE, recall não-zero, DTW
└── utils/
    ├── paths.py         # constantes (CD_MUN dos 23 munis, DISEASES, dirs — CCD_CACHE_DIR)
    └── splits.py        # rolling origin
notebooks/
├── train_all.py         # orquestra baselines + os 5 deep models
├── ensemble.py          # top-3 por doença × horizonte
├── forecast_plots.py    # figuras de diagnóstico
└── README.md            # como rodar
scripts/
└── make_results.py      # consolida reports/ -> results/tables + results/figures
run_deepar_only.py        # roda só o DeepAR (útil para re-runs isolados, ex. Colab)
run_experiments.sh        # pipeline completo, ponta a ponta
```

## 8. Setup

```bash
conda create -n ccd python=3.11 -y
conda activate ccd
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy pyarrow openpyxl odfpy python-calamine scikit-learn matplotlib seaborn \
            lightning tensorboard pytorch-forecasting statsforecast lightgbm catboost xgboost \
            prophet fastdtw optuna pyyaml jupyter nbformat ipykernel
python -m ipykernel install --user --name ccd --display-name "Python (ccd)"
```

**Dados:** os arquivos `*.xlsx`, `*.ods`, `*.csv` **não estão versionados** — coloque-os
na raiz e em `Taxas/` antes de rodar (ver formato em §2).

**Cache:** parquet derivado dos Excel vai para o diretório apontado pela variável de
ambiente `CCD_CACHE_DIR` (portável entre Linux/Windows/Colab); se não definida, usa um
diretório temporário padrão do sistema.

## 9. Como rodar

**Opção rápida — pipeline completo de ponta a ponta:**
```bash
bash run_experiments.sh
```
Instala as deps que faltarem, monta o painel, roda os 12 modelos (7 baselines + 5 deep)
em rolling-origin, e gera as tabelas/figuras em `results/`.

**Opção manual — notebook (edição/depuração interativa):**
1. Abrir `notebooks/train_all.ipynb` (gerado a partir de `train_all.py` via `build_notebook.py`).
2. Selecionar kernel **Python (ccd)**.
3. **Run All**.

Cada (modelo × doença) cacheia em `reports/deep_<modelo>_<doenca>.csv`. Para forçar
re-treino, apague o CSV correspondente. Para rodar só um modelo específico (ex.: depois
de corrigir algo), veja `run_deepar_only.py` como referência de script standalone.

**Variância entre sementes:** rode `python -m src.models.deep_panel <doenca> <modelo> 42,1,7`
para treinar com múltiplas sementes; `make_results.py` detecta a coluna `seed` e
preenche `mae_std_seed` automaticamente.

## 10. Próximos passos

- **Importância de covariáveis nos deep models** — rodar `python -m src.models.deep_panel <doenca> tft 42`
  em GPU (as árvores já estão em `results/tables/covariate_importance/`).
- **ZINB** explícito para hanseníase/hepatite/sífilis.
- **Hyperparameter search** com Optuna (hidden_size, dropout, lr, encoder_length).
- **Conformal prediction** post-hoc para intervalos calibrados.
- **Re-rodar o ensemble** (`notebooks/ensemble.py`) com o conjunto completo de 12 modelos
  (o `results/ensemble/` atual é de uma configuração anterior — ver aviso no README de lá).
