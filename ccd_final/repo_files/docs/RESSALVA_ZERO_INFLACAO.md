# Ressalva importante: a armadilha da zero-inflação

**Este documento registra uma limitação que afeta a interpretação do resultado principal.
Deve ser lido junto com `results/README.md` antes de citar os vencedores por doença.**

## O achado

O ranking por MAE, sozinho, é enganoso nas doenças fortemente zero-infladas. Cruzando o MAE
com o recall de eventos não-zero (proporção dos meses-município com casos reais que o modelo
sinalizou), aparece uma relação monotônica com a fração de zeros da série:

| doença | zero-fraction (mediana entre municípios) | vencedor por MAE | MAE | **recall do vencedor** |
|---|---|---|---|---|
| hanseníase | 88% | nhits | 0,357 | **0,185** |
| hepatite | 70% | lstm | 0,189 | **0,450** |
| sífilis | 41% | xgboost | 2,651 | 0,932 |
| tuberculose | 14% | seasonal_ma3 | 2,773 | 0,949 |
| HIV/aids | 8% | catboost | 2,496 | 1,000 |

Quanto mais zerada a série, menos o "melhor modelo" detecta os eventos que importam.

## Caso mais grave: hanseníase

O `nhits` vence em MAE (0,357), mas suas previsões são praticamente nulas:

- `y_pred`: mínimo **8,4 × 10⁻²²**, média **0,051**, máximo 1,64
- `y_true`: média 0,389, máximo 20

Ou seja: o modelo aprendeu a prever ~zero sempre. Com 88% de meses sem casos, isso rende um
MAE excelente e um recall de apenas 18,5% — o modelo é ótimo na métrica e inútil na prática
epidemiológica, porque não sinaliza os surtos.

Isso é visível na figura `results/figures/forecasts/forecast_hanseniase_nhits.png`: as linhas
de previsão ficam coladas no eixo x, praticamente invisíveis contra a série observada.

Comparação direta, mesma doença:

| modelo | MAE | recall | leitura |
|---|---|---|---|
| nhits | 0,357 | 0,185 | melhor MAE, quase não detecta nada |
| lstm / gru | 0,358 / 0,360 | 0,209 | mesmo comportamento |
| **xgboost** | 0,419 | **0,580** | MAE 17% pior, detecta **3× mais** eventos |
| deepar | 0,929 | **0,710** | pior MAE, melhor detecção (superprevê) |

## Implicação para o artigo

Não reportar o vencedor por doença apenas por MAE. Três formas de tratar:

1. **Reportar MAE e recall lado a lado** e discutir o trade-off explicitamente — a tabela
   `results/tables/final_summary.csv` já traz as duas colunas.
2. **Escolher o modelo por doença conforme o objetivo**: se o uso é vigilância (detectar
   surtos), o `xgboost` é preferível ao `nhits` em hanseníase, apesar do MAE pior.
3. **Justificar o ZINB no roadmap**: a zero-inflação extrema de hanseníase/hepatite é
   exatamente o caso em que uma verossimilhança zero-inflada explícita se justifica — o
   roadmap já prevê isso e este achado dá o argumento empírico.

Figura de apoio: `results/figures/comparison/mae_vs_recall_tradeoff.png`.

## Anomalia adicional: GRU em hepatite, h=12

O `gru` em hepatite salta de MAE 0,167 (h=1) para **1,046** (h=12), enquanto os demais ficam
em torno de 0,2 — é o pico visível em `comparison_mae_by_horizon.png`. O recall acompanha o
salto (0,2 → 0,9): no horizonte longo o modelo inverte o comportamento, passando de
subprevisão para superprevisão. Não é erro de plotagem; é instabilidade real do modelo nesse
horizonte, e vale uma nota no texto caso o GRU seja discutido.
