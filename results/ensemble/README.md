# Ensemble — top-3 por (doença × horizonte)

**Split:** validação 2019-2020 (escolhe top-3 + pesos), teste 2021-2022 (avalia).

## Estratégias

- `top3_mean` — média simples dos 3 melhores na validação
- `top3_inv_mae` — média ponderada por 1/MAE
- `all_mean` — sanidade: média de todos os modelos

## MAE médio no teste

| model        |   mae |
|:-------------|------:|
| all_mean     | 1.78  |
| top3_inv_mae | 1.695 |
| top3_mean    | 1.755 |

## Ensemble vs melhor single

| disease     |   horizon | best_model   |   mae_best_single |   mae_ensemble |   delta_mae | ensemble_wins   |
|:------------|----------:|:-------------|------------------:|---------------:|------------:|:----------------|
| hepatite    |         1 | tft          |             0.096 |          0.128 |       0.032 | False           |
| hanseniase  |         1 | tft          |             0.144 |          0.189 |       0.046 | False           |
| hanseniase  |         3 | nhits        |             0.17  |          0.253 |       0.083 | False           |
| hepatite    |         3 | tft          |             0.207 |          0.294 |       0.086 | False           |
| hepatite    |        12 | nhits        |             0.212 |          0.245 |       0.032 | False           |
| hanseniase  |        12 | nhits        |             0.213 |          0.226 |       0.013 | False           |
| hepatite    |         6 | lgbm         |             0.272 |          0.286 |       0.014 | False           |
| hanseniase  |         6 | tft          |             0.356 |          0.301 |      -0.054 | True            |
| hiv_aids    |         6 | lgbm         |             1.872 |          2.374 |       0.502 | False           |
| hiv_aids    |         1 | tft          |             1.922 |          2.301 |       0.379 | False           |
| hiv_aids    |         3 | sarima       |             2.003 |          2.155 |       0.151 | False           |
| hiv_aids    |        12 | deepar       |             2.106 |          2.176 |       0.07  | False           |
| sifilis     |         6 | tft          |             2.216 |          2.315 |       0.1   | False           |
| sifilis     |         1 | deepar       |             2.477 |          2.634 |       0.157 | False           |
| tuberculose |        12 | deepar       |             2.552 |          2.439 |      -0.113 | True            |
| sifilis     |         3 | sarima       |             2.663 |          3.057 |       0.394 | False           |
| tuberculose |         6 | deepar       |             2.68  |          2.903 |       0.223 | False           |
| tuberculose |         1 | catboost     |             2.777 |          2.981 |       0.204 | False           |
| tuberculose |         3 | lgbm         |             2.962 |          3.045 |       0.084 | False           |
| sifilis     |        12 | sarima       |             3.528 |          3.605 |       0.077 | False           |

Ensemble vence em **2/20** pares.
