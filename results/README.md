# Resultados — predição multi-doença em 23 municípios de SP

Síntese consolidada do pipeline: baselines (SeasonalNaive, SeasonalMA3, SARIMA, LightGBM, CatBoost) + deep panel (TFT, N-HiTS, DeepAR).

## Resumo agregado (MAE médio entre origens)

| model          |    1 |    3 |    6 |   12 |
|:---------------|-----:|-----:|-----:|-----:|
| catboost       | 1.82 | 1.82 | 1.98 | 1.87 |
| deepar         | 2.04 | 2.3  | 1.97 | 2.04 |
| lgbm           | 1.81 | 1.74 | 1.79 | 1.86 |
| nhits          | 2.48 | 2.32 | 2.03 | 2.5  |
| sarima         | 1.8  | 1.76 | 1.78 | 2.05 |
| seasonal_ma3   | 1.81 | 1.88 | 2.08 | 1.94 |
| seasonal_naive | 1.99 | 2.22 | 2.14 | 2.26 |
| tft            | 1.83 | 2.03 | 1.96 | 1.87 |

## Ranking por (doença × horizonte)

| disease     |   horizon | 1st                 | 2nd                   | 3rd                 |
|:------------|----------:|:--------------------|:----------------------|:--------------------|
| hanseniase  |         1 | nhits (0.17)        | tft (0.24)            | lgbm (0.35)         |
| hanseniase  |         3 | nhits (0.18)        | lgbm (0.37)           | catboost (0.41)     |
| hanseniase  |         6 | nhits (0.36)        | tft (0.41)            | lgbm (0.52)         |
| hanseniase  |        12 | nhits (0.21)        | lgbm (0.35)           | catboost (0.39)     |
| hepatite    |         1 | tft (0.10)          | nhits (0.10)          | catboost (0.21)     |
| hepatite    |         3 | tft (0.18)          | nhits (0.22)          | catboost (0.26)     |
| hepatite    |         6 | lgbm (0.24)         | sarima (0.26)         | catboost (0.27)     |
| hepatite    |        12 | tft (0.23)          | seasonal_naive (0.23) | nhits (0.24)        |
| hiv_aids    |         1 | tft (2.21)          | deepar (2.45)         | sarima (2.52)       |
| hiv_aids    |         3 | sarima (2.16)       | catboost (2.29)       | tft (2.35)          |
| hiv_aids    |         6 | deepar (2.29)       | nhits (2.53)          | sarima (2.92)       |
| hiv_aids    |        12 | deepar (2.16)       | catboost (2.48)       | lgbm (2.63)         |
| sifilis     |         1 | sarima (2.59)       | lgbm (3.02)           | seasonal_ma3 (3.03) |
| sifilis     |         3 | sarima (2.80)       | seasonal_ma3 (2.84)   | lgbm (2.90)         |
| sifilis     |         6 | sarima (2.31)       | tft (2.49)            | lgbm (2.62)         |
| sifilis     |        12 | lgbm (3.24)         | sarima (3.29)         | seasonal_ma3 (3.41) |
| tuberculose |         1 | seasonal_ma3 (2.42) | catboost (2.66)       | lgbm (2.85)         |
| tuberculose |         3 | lgbm (2.69)         | sarima (2.93)         | seasonal_ma3 (2.99) |
| tuberculose |         6 | catboost (2.61)     | lgbm (2.65)           | sarima (2.69)       |
| tuberculose |        12 | tft (2.40)          | seasonal_ma3 (2.67)   | deepar (2.70)       |

## Vencedor por doença (MAE médio entre horizontes)

| disease     | model   |   mae |
|:------------|:--------|------:|
| hanseniase  | nhits   |  0.23 |
| hepatite    | tft     |  0.22 |
| hiv_aids    | deepar  |  2.46 |
| sifilis     | sarima  |  2.75 |
| tuberculose | lgbm    |  2.75 |

## Tabelas por município

Uma tabela por doença em [`tables/per_muni_mae_<doenca>.csv`](tables/) — MAE de cada modelo deep para cada um dos 23 municípios. Heatmaps em [`figures/comparison/`](figures/comparison/).

## Figuras

- **EDA**: `figures/eda/` — zero-fraction, séries por município, correlação clima×alvo, rolling origin
- **Forecasts**: `figures/forecasts/` — predito vs observado, grade 6×4 por (doença × modelo deep)
- **Comparação**: `figures/comparison/` — MAE por horizonte, estabilidade por origem, heatmap por município
