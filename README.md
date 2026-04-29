# ccd_fin — predição mensal de doenças em 23 municípios de SP

Pipeline de previsão temporal multi-doença (hanseníase, hepatite, HIV/AIDS, sífilis, tuberculose) em 23 municípios do estado de São Paulo, com features climáticas e socioeconômicas. Implementado em PyTorch com `pytorch-forecasting` + Lightning, comparado contra baselines clássicos (SARIMA, LightGBM).

---

## 1. Objetivo

Predizer a **incidência mensal** das 5 doenças nos 23 municípios-alvo (Barueri, Bauru, Campinas, Carapicuíba, Diadema, Guarujá, Guarulhos, Itapevi, Jundiaí, Mauá, Osasco, Paulínia, Praia Grande, Ribeirão Preto, Santo André, Santos, São Bernardo do Campo, São José do Rio Preto, São José dos Campos, São Paulo, São Vicente, Sorocaba, Taboão da Serra) em horizontes de **1, 3, 6 e 12 meses**.

## 2. Dados

| Grupo | Granularidade | Período | Fonte |
|---|---|---|---|
| **Doenças (contagens + taxas)** | mensal | 2000-01 → 2023-12 (288 meses) | `<doenca>_00_23.xlsx` + `Taxas/TX_*.xlsx` |
| **Clima** (evapot, precip, T_min, T_max, umid) | mensal | 1999-01 → 2023-12 | 5 arquivos `Evapot_SP.xlsx`, `Precip_SP.xlsx`, etc. |
| **Socioeconômico** (pop, dens. demog., PPC, urbanização) | anual | 2000 → 2023 | `Pop_Geral_SP.xlsx`, `Dens_demog_SP.csv`, etc. |

**645 municípios de SP** em todos os arquivos (recortamos os 23 alvos).

**Painel final:** 23 munis × 288 meses = **6.624 linhas × 22 colunas**, 0% missing após backfill da taxa de hanseníase/tuberculose para 2023 (que faltava nos arquivos `TX_*`) usando `n / pop × 100k`.

## 3. Descobertas da EDA

### 3.1 Zero-inflation severa em algumas doenças

Fração de meses com 0 casos reportados (mediana entre os 23 munis):

| Doença | Zero-fraction (mediana / máx) | Var/Mean | Loss recomendada |
|---|---|---|---|
| **hanseníase** | **88% / 97%** | 1.2 | ZINB (zero-inflated NB) |
| hepatite | 70% / 93% | 2.4 | ZINB |
| sífilis | 41% / 88% | 2.7 | ZINB |
| tuberculose | 14% / 41% | 3.0 | NegativeBinomial |
| HIV/aids | 8% / 61% | 3.8 | NegativeBinomial |

**Implicações:**
- Hanseníase tem ~88% de zeros — a previsão é dominada pela classificação "vai ter caso ou não". Modelos de regressão padrão (MAE/MSE) viesam tudo pra zero.
- Sobredispersão (var/média ≫ 1) em todas as doenças → **Poisson é insuficiente**, NB é a escolha mínima.
- Estratégia adotada: predizer **contagem** (`n_<doenca>`) e converter pra taxa via `n / pop × 100k`. Loss alinhada ao perfil de zeros.

### 3.2 Clima tem correlação fraca com incidência

Spearman médio entre `clima(t-k)` e `taxa(t)` para `k ∈ [0..6]`: **|ρ| < 0.08** em quase todos os pares. Umidade é o sinal mais consistente (ρ ~ 0.05-0.08, lag 3-6) com hepatite e HIV.

**Implicação:** features climáticas são incluídas nos modelos, mas **não esperamos ganho grande** delas em isolado. Modelos não-lineares (TFT, GBM) podem extrair sinal residual via interações.

### 3.3 Choque COVID (2020-03 → 2021-12)

Reporting de doenças cai abruptamente nesse período em quase todos os munis (visível nas séries). Tratado como flag (`covid_period`) que entra como feature; também avaliamos rolling origin **com** e **sem** COVID na janela de teste.

## 4. Design experimental

- **Alvo:** contagens mensais (`n_<doenca>`); taxas derivadas para visualização.
- **Split temporal:** **rolling origin** com 4 origens (`2019-12`, `2020-12`, `2021-12`, `2022-12`). Treina até a origem, testa nos próximos 12 meses; computa métricas por horizonte e por origem (mostra estabilidade vs choque COVID).
- **Métricas:** MAE, RMSE, sMAPE — agregados por (modelo × doença × horizonte) e por origem.
- **Treino:** apenas nos 23 municípios-alvo. Loss adequada à doença (NB padrão; ZINB seria a próxima iteração).

## 5. Modelos implementados

### Tier 1 — Baselines (`src/models/baselines.py`)
- **SeasonalNaive**: `y_hat[t+h] = y[t+h-12]`
- **SeasonalMA3**: média dos mesmos meses dos últimos 3 anos
- **AutoARIMA / SARIMA** via `statsforecast` (sazonalidade 12)
- **LightGBM Poisson** com features: lags do alvo (1, 2, 3, 6, 12, 13), rolling means, lags de clima (1, 3, 12), embedding de município + mês como categóricos
- **CatBoost Poisson** mesmas features; vence em alguns horizontes médios/longos (hepatite h=6/12, sífilis h=12, tuberculose h=3)

### Tier 2 — Deep panel (`src/models/deep_panel.py`)
- **TFT** (`TemporalFusionTransformer`) com `QuantileLoss` — covariáveis estáticas (muni + socio), temporais (clima + calendário), atenção interpretável
- **N-HiTS** (`NHiTS`) com `QuantileLoss` — backbone hierárquico de blocos MLP
- **DeepAR** com `NegativeBinomialDistributionLoss` — RNN autorregressivo com saída probabilística NB nativa (alinhado à sobredispersão observada na EDA)

Todos com **EarlyStopping** (patience=15) e **ReduceLROnPlateau** (patience=5). `max_epochs=200` é só teto; convergência típica entre 40-100 épocas.

> **Nota técnica sobre DeepAR + NB:** a transformação `softplus` no `GroupNormalizer` (default em panels com NB loss) degenera em séries com >50% de zeros e produz NaN nas predições — bug numérico conhecido em `pytorch-forecasting 1.7`. Substituímos por `log1p`, que é estável em contagens esparsas.

### Não implementado (decisões e razões)
- **ZINB explícito:** EDA confirma que valeria pra hanseníase/hepatite/sífilis, mas não chegamos a implementar — fica no roadmap.
- **ST-GNN:** com 23 nós, ganho marginal vs custo de integração (`torch-geometric-temporal` tem wheels frágeis no Windows). Decidido pular.

## 6. Resultados (todos os modelos)

Médias agregadas (5 doenças × 4 origens, MAE em contagem):

| Modelo | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| **sarima** | **1.80** | 1.76 | **1.78** | 2.05 |
| **lgbm** | 1.81 | **1.74** | 1.79 | **1.86** |
| seasonal_ma3 | 1.81 | 1.88 | 2.08 | 1.94 |
| catboost | 1.82 | 1.82 | 1.98 | **1.87** |
| tft | 1.83 | 2.03 | 1.96 | 1.87 |
| seasonal_naive | 1.99 | 2.22 | 2.14 | 2.26 |
| deepar | 2.04 | 2.30 | 1.97 | 2.04 |
| nhits | 2.48 | 2.32 | 2.03 | 2.50 |

**Melhor modelo por doença** (MAE médio entre horizontes):

| Doença | Vencedor | Observação |
|---|---|---|
| hanseníase | **N-HiTS** | aprende padrão de quase-tudo-zero melhor que GBM |
| hepatite | **TFT** | covariáveis temporais ajudam |
| HIV/aids | **DeepAR** | NB nativa modela bem a sobredispersão |
| sífilis | **SARIMA** | série bem-sazonal, modelos clássicos vencem |
| tuberculose | empate (LGBM/SARIMA/CatBoost) | sinal mais limpo, todos performam similar |

**Lições:**
- Cada doença tem um vencedor diferente — **não há "melhor modelo único"**, ensemble seria a próxima evolução.
- Deep models superam GBM em 3 das 5 doenças — vale o custo de treino.
- Naive sazonal é piso decente em h=1 para doenças com muitos zeros — útil como sanidade.
- Rolling origin sobre 2019-2022 mostra degradação clara em 2020-2021 (choque COVID).

Tabelas detalhadas e figuras em [`results/`](results/).

## 7. Estrutura do repo

```
src/
├── data/
│   ├── load.py          # leitores wide→long (xlsx strict OOXML, ods, csv)
│   ├── build_panel.py   # painel unificado 23 munis × 288 meses
│   └── eda.py           # zero-fraction, missingness, correlação clima×alvo
├── models/
│   ├── baselines.py     # SeasonalNaive, SARIMA, LightGBM
│   └── deep_panel.py    # TFT, N-HiTS via pytorch-forecasting
├── eval/
│   └── metrics.py       # MAE, RMSE, sMAPE, MAPE, WQL
└── utils/
    ├── paths.py         # constantes (CD_MUN dos 23 munis, DISEASES, dirs)
    └── splits.py        # rolling origin
notebooks/
├── train_all.ipynb      # pipeline end-to-end com diagnóstico visual
├── train_all.py         # fonte editável (cells # %%)
├── build_notebook.py    # converte .py → .ipynb
└── README.md            # como rodar
```

## 8. Setup

```bash
conda create -n ccd python=3.11 -y
conda activate ccd
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy pyarrow openpyxl odfpy python-calamine scikit-learn matplotlib seaborn \
            lightning tensorboard pytorch-forecasting statsforecast lightgbm optuna pyyaml \
            jupyter nbformat ipykernel
python -m ipykernel install --user --name ccd --display-name "Python (ccd)"
```

**Dados:** os arquivos `*.xlsx`, `*.ods`, `*.csv` **não estão versionados** — coloque-os na raiz e em `Taxas/` antes de rodar (ver formato em §2).

**Cache:** parquet derivado dos Excel vai pra `C:\temp\ccd_cache\` (fora do OneDrive — recomendado para evitar sync de artefatos pesados).

## 9. Como rodar

1. Abrir `notebooks/train_all.ipynb` no VS Code.
2. Selecionar kernel **Python (ccd)**.
3. **Run All** (~1h na GPU para o pipeline completo).

Cada (modelo × doença) cacheia em `reports/deep_<modelo>_<doenca>.csv`. Para forçar re-treino, apague os CSVs.

Saídas:
- `reports/baselines.csv`, `reports/final_summary.csv` — métricas
- `reports/figures/*.png` — heatmaps, séries por muni, comparações, calibração

## 10. Próximos passos

- **ZINB** explícito para hanseníase/hepatite/sífilis (espera-se ganho substancial).
- **Hyperparameter search** com Optuna (hidden_size, dropout, lr, encoder_length).
- **Ensemble** dos top-3 modelos por doença/horizonte.
- **Conformal prediction** post-hoc para intervalos calibrados.
- **Tuning de DeepAR**: testar `hidden_size` maior (32-64) e mais épocas — first run com hidden=24/40ep ainda perde do LGBM em horizontes curtos.
