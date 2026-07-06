# Resultados — predição multi-doença em 23 municípios de SP

Síntese consolidada do pipeline: baselines (SeasonalNaive, SeasonalMA3, SARIMA, LightGBM, CatBoost, XGBoost, Prophet) + deep panel (TFT, N-HiTS, DeepAR, LSTM, GRU).

`final_summary.csv` traz, por modelo × doença × horizonte: MAE, RMSE, R², SMAPE, recall/precisão de eventos não-zero, e desvio-padrão do MAE entre origens (`mae_std_origin`) e entre sementes (`mae_std_seed`, preenchido só em runs multi-seed). DTW por modelo × doença em `dtw_by_model.csv`.

## Resumo agregado (MAE médio entre origens)

| model          |    1 |    3 |    6 |   12 |
|:---------------|-----:|-----:|-----:|-----:|
| catboost       | 1.9  | 1.76 | 1.66 | 1.8  |
| deepar         | 1.87 | 1.93 | 2.02 | 1.94 |
| gru            | 1.86 | 1.89 | 2.27 | 2.26 |
| lgbm           | 1.79 | 1.76 | 1.69 | 1.92 |
| lstm           | 2.03 | 1.92 | 2.23 | 2.02 |
| nhits          | 2.25 | 2.16 | 2.06 | 2.47 |
| prophet        | 1.95 | 1.79 | 1.91 | 1.95 |
| sarima         | 1.8  | 1.76 | 1.79 | 2.05 |
| seasonal_ma3   | 1.81 | 1.88 | 2.08 | 1.94 |
| seasonal_naive | 1.99 | 2.22 | 2.14 | 2.26 |
| tft            | 1.82 | 1.9  | 2.15 | 2.14 |
| xgboost        | 1.73 | 1.68 | 1.62 | 1.95 |

## Ranking por (doença × horizonte)

| disease     |   horizon | 1st                 | 2nd             | 3rd                   |
|:------------|----------:|:--------------------|:----------------|:----------------------|
| hanseniase  |         1 | gru (0.18)          | lstm (0.20)     | nhits (0.25)          |
| hanseniase  |         3 | nhits (0.31)        | lstm (0.36)     | gru (0.38)            |
| hanseniase  |         6 | xgboost (0.48)      | nhits (0.48)    | lstm (0.50)           |
| hanseniase  |        12 | gru (0.37)          | lstm (0.38)     | nhits (0.39)          |
| hepatite    |         1 | lstm (0.14)         | prophet (0.15)  | nhits (0.17)          |
| hepatite    |         3 | lstm (0.17)         | gru (0.19)      | prophet (0.21)        |
| hepatite    |         6 | nhits (0.20)        | tft (0.24)      | lstm (0.24)           |
| hepatite    |        12 | lstm (0.20)         | prophet (0.22)  | seasonal_naive (0.23) |
| hiv_aids    |         1 | tft (2.50)          | sarima (2.52)   | deepar (2.64)         |
| hiv_aids    |         3 | sarima (2.16)       | xgboost (2.16)  | catboost (2.21)       |
| hiv_aids    |         6 | catboost (2.23)     | lgbm (2.33)     | xgboost (2.33)        |
| hiv_aids    |        12 | deepar (2.21)       | catboost (2.50) | seasonal_ma3 (2.93)   |
| sifilis     |         1 | xgboost (2.47)      | sarima (2.58)   | deepar (2.65)         |
| sifilis     |         3 | prophet (2.42)      | lgbm (2.66)     | xgboost (2.68)        |
| sifilis     |         6 | xgboost (2.32)      | sarima (2.33)   | catboost (2.48)       |
| sifilis     |        12 | lgbm (2.93)         | catboost (2.97) | xgboost (3.13)        |
| tuberculose |         1 | seasonal_ma3 (2.42) | gru (2.47)      | prophet (2.53)        |
| tuberculose |         3 | prophet (2.79)      | xgboost (2.85)  | catboost (2.92)       |
| tuberculose |         6 | deepar (2.51)       | sarima (2.69)   | xgboost (2.70)        |
| tuberculose |        12 | tft (2.63)          | gru (2.67)      | seasonal_ma3 (2.67)   |

## Vencedor por doença (MAE médio entre horizontes)

| disease     | model        |   mae |
|:------------|:-------------|------:|
| hanseniase  | nhits        |  0.36 |
| hepatite    | lstm         |  0.19 |
| hiv_aids    | catboost     |  2.5  |
| sifilis     | xgboost      |  2.65 |
| tuberculose | seasonal_ma3 |  2.77 |

## Tabelas por município

Uma tabela por doença em [`tables/per_muni_<doenca>.csv`](tables/) — **MAE, RMSE, R² e SMAPE** de cada modelo (deep + baselines) para cada um dos 23 municípios. Matrizes wide de MAE/RMSE em `tables/per_muni_mae_<doenca>.csv` / `per_muni_rmse_<doenca>.csv`; heatmaps em [`figures/comparison/`](figures/comparison/).

> R² é NaN para séries de alvo constante (municípios com quase-tudo-zero, ex. hanseníase) — esperado, não é erro.

## Figuras

- **EDA**: `figures/eda/` — zero-fraction, séries por município, correlação clima×alvo, rolling origin
- **Forecasts**: `figures/forecasts/` — predito vs observado, grade 6×4 por (doença × modelo deep)
- **Comparação**: `figures/comparison/` — MAE por horizonte, estabilidade por origem, heatmap por município
