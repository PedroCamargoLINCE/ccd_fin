# Ensemble — top-3 por (doença × horizonte)

**Split:** validação 2019-2020 (escolhe top-3 + pesos), teste 2021-2022 (avalia).

## Estratégias

- `top3_mean` — média simples dos 3 melhores na validação
- `top3_inv_mae` — média ponderada por 1/MAE
- `all_mean` — sanidade: média de todos os modelos

## MAE médio no teste

| model        |   mae |
|:-------------|------:|
| all_mean     | 1.717 |
| top3_inv_mae | 1.749 |
| top3_mean    | 1.749 |

## Ensemble vs melhor single

| disease     |   horizon | best_model     |   mae_best_single |   mae_ensemble |   delta_mae | ensemble_wins   |
|:------------|----------:|:---------------|------------------:|---------------:|------------:|:----------------|
| hanseniase  |         1 | gru            |             0.069 |          0.087 |       0.018 | False           |
| hepatite    |         1 | lstm           |             0.072 |          0.089 |       0.017 | False           |
| hepatite    |        12 | seasonal_naive |             0.239 |          0.292 |       0.053 | False           |
| hanseniase  |         3 | nhits          |             0.249 |          0.37  |       0.121 | False           |
| hepatite    |         3 | lstm           |             0.251 |          0.283 |       0.031 | False           |
| hanseniase  |        12 | nhits          |             0.263 |          0.351 |       0.089 | False           |
| hepatite    |         6 | nhits          |             0.265 |          0.291 |       0.027 | False           |
| hanseniase  |         6 | gru            |             0.562 |          0.638 |       0.076 | False           |
| hiv_aids    |         6 | lgbm           |             1.74  |          1.834 |       0.094 | False           |
| hiv_aids    |         3 | xgboost        |             1.774 |          2.352 |       0.578 | False           |
| hiv_aids    |         1 | seasonal_ma3   |             1.978 |          2.66  |       0.682 | False           |
| tuberculose |        12 | gru            |             2.13  |          2.536 |       0.406 | False           |
| hiv_aids    |        12 | deepar         |             2.222 |          2.523 |       0.301 | False           |
| sifilis     |         3 | prophet        |             2.425 |          2.606 |       0.181 | False           |
| sifilis     |         1 | prophet        |             2.523 |          2.622 |       0.098 | False           |
| sifilis     |         6 | tft            |             2.555 |          2.582 |       0.027 | False           |
| tuberculose |         6 | prophet        |             2.599 |          3.028 |       0.429 | False           |
| sifilis     |        12 | lgbm           |             2.603 |          3.347 |       0.744 | False           |
| tuberculose |         1 | seasonal_ma3   |             2.942 |          3.035 |       0.093 | False           |
| tuberculose |         3 | lstm           |             3.02  |          3.461 |       0.441 | False           |

Ensemble vence em **0/20** pares.
