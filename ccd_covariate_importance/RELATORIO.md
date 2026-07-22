# Importância de covariáveis — análise real + código para runs futuras

## Por que isso não estava pronto antes (explicação direta)

Importância de covariável precisa do **objeto do modelo treinado** (`feature_importances_`,
`get_feature_importance()`, `interpret_output()`) — não dá pra extrair só das predições em
CSV. Nem o run de GPU do PR #3 nem o Colab do DeepAR salvaram os modelos treinados em lugar
acessível: o `baselines.py` treinava e descartava o modelo assim que gerava a predição, e os
checkpoints dos deep models ficaram em máquinas efêmeras (portal GPU, sessão do Colab já
encerrada). Eu sinalizei isso como "pendente" em vários pacotes, mas isso foi adiar o
problema em vez de resolvê-lo — o certo era ter embutido a extração no código de treino desde
o início. Consertado agora em duas frentes:

1. **Modelos de árvore (LightGBM, CatBoost, XGBoost):** re-treinei localmente com o esquema
   as-of-origem já corrigido (mesma lógica do `baselines.py` atual, sem GPU necessária — são
   rápidos), extraindo a importância real a cada fit. **Isso já está pronto e analisado
   abaixo — não é uma promessa, é resultado real.**
2. **TFT:** o único dos 5 deep models com importância de variável nativa
   (`interpret_output`, variable selection network). Testei a extração de ponta a ponta aqui
   (treino real, 2 épocas, CPU) e o código já está embutido no `deep_panel.py` — **a próxima
   run em GPU já vai gerar isso sozinha**, sem passo manual. Não retreinei os 5 deep models
   completos aqui (precisaria de GPU pro treino cheio); o que testei foi a mecânica de
   extração, que funciona.

## Resultado 1 — Importância por grupo de covariável (árvores, real, 3 modelos × 5 doenças × todas origens/horizontes)

Proporção média da importância (gain) capturada por cada grupo de covariável:

| doença | alvo (lags/rolling) | clima (defasado) | socioeconômico | sazonalidade (mês) | município |
|---|---|---|---|---|---|
| hanseníase | 47,0% | 33,5% | 13,4% | 0,9% | 5,2% |
| hepatite | 58,0% | 27,3% | 11,1% | 0,8% | 2,8% |
| hiv_aids | 68,0% | 19,5% | 9,5% | 0,7% | 2,3% |
| sífilis | 55,0% | 27,0% | 14,4% | 1,0% | 2,7% |
| tuberculose | 60,1% | 21,5% | 13,2% | 0,8% | 4,5% |

O próprio histórico da doença (lags e médias móveis) domina em todas — esperado pra série
temporal. Clima aparece com peso substancial (19-33%), maior que o esperado pela correlação
Spearman fraca vista na EDA (|ρ|<0,08) — modelos de árvore capturam interação não-linear que
correlação simples não vê.

## Resultado 2 — Top features individuais por doença

Ex.: tuberculose → `n_tuberculose_r3` (16,7%), `n_tuberculose_r6` (15,5%), `n_tuberculose_r12`
(11,0%), `n_tuberculose_l0` (6,3%), `ppc` (5,2%). Padrão parecido nas outras 4 doenças — média
móvel de 3-12 meses do próprio alvo lidera, seguida do valor mais recente (`l0`/`l1`) e do PPC
(indicador socioeconômico). Tabela completa em `feature_importance_top_features.csv`.

## Resultado 3 — Ablação: clima ajuda ou atrapalha? (o ponto mais importante)

**Importância de gain ≠ utilidade real fora da amostra.** Uma covariável pode ter peso positivo
no modelo e ainda assim piorar a generalização (ruído/overfitting). Rodei uma ablação de
verdade: LightGBM **com** clima vs. **sem** clima, MAE real em teste, mesmo esquema
as-of-origem:

| doença | MAE com clima | MAE sem clima | Δ MAE | clima ajuda? |
|---|---|---|---|---|
| hanseníase | 0,423 | 0,438 | +0,015 | **sim** |
| hepatite | 0,264 | 0,255 | −0,009 | **não** |
| hiv_aids | 2,589 | 2,665 | +0,076 | **sim** |
| sífilis | 2,811 | 2,732 | −0,079 | **não** |
| tuberculose | 2,757 | 2,742 | −0,015 | **não** |

Achado real e nuançado: clima **ajuda de fato** só em hanseníase e HIV/aids. Nas outras três
(hepatite, sífilis, tuberculose), **remover o clima melhora o MAE** — ou seja, o modelo está
"usando" essas variáveis (importância de gain positiva na tabela acima) mas isso está
prejudicando a generalização, não ajudando. É exatamente o "quais atrapalharam" que a revisão
pediu, e a resposta não é uniforme entre doenças — vale reportar assim no texto, não como uma
conclusão única de "clima ajuda" ou "clima não ajuda".

Quebra por horizonte em `ablation_climate_by_horizon.csv` — o efeito também não é constante
dentro da mesma doença (ex.: sífilis h=12 tem Δ=−0,23, bem mais negativo que a média da
doença), então cuidado ao generalizar até por horizonte.

## O que ficou de fora (limitação honesta)

- **Importância do TFT/N-HiTS/DeepAR/LSTM/GRU com números reais** — não retreinei os 5 deep
  models completos aqui (precisa GPU pra ser viável em tempo razoável). O código de extração
  pro TFT está pronto e testado; falta só a próxima run em GPU gerar os números de verdade.
  N-HiTS/DeepAR/LSTM/GRU **não têm** importância de variável nativa equivalente ao TFT — pra
  esses, a única forma de "importância" seria uma ablação como a que fiz acima (retreinar
  com/sem grupo de covariável), o que dá pra fazer mas não fiz para os 5 ainda.
- **Ablação só com LightGBM** — usei 1 modelo como representante rápido pra cobrir as 5
  doenças em tempo hábil. Catboost/XGBoost provavelmente concordam (a importância de gain é
  parecida entre os 3), mas não testei a ablação neles individualmente.

## Código adicionado (drop-in, testado)

| Arquivo | O que mudou |
|---|---|
| `src/models/baselines.py` | `_tree_asof`, `_fp_lgbm/_fp_catboost/_fp_xgboost` agora retornam também a importância do fit; `run_all_baselines` ganha `save_importance_path` — a próxima run salva `reports/feature_importance_trees_raw.csv` sozinha |
| `src/models/deep_panel.py` | `extract_tft_importance()` novo; `run_deep_single`/`run_deep_multiseed` ganham `imp_records`; rodando `python -m src.models.deep_panel <doenca> tft <seeds>` já salva `reports/feature_importance_tft_<doenca>.csv` sozinho |

Testei os dois com fits reais (LightGBM/CatBoost/XGBoost re-treinados aqui; TFT com 2 épocas
em CPU) — sem erros, formato correto.

## Como aplicar

```bash
cd ~/Documents/ccd_fin_clone/ccd_fin
git pull
unzip -o ccd_covariate_importance.zip
cp -r ccd_covariate_importance/repo_files/* .
mkdir -p results/tables/covariate_importance
cp ccd_covariate_importance/analysis/*.csv results/tables/covariate_importance/

git add -A
git commit -m "feat: extracao de importancia de covariaveis (arvores real + TFT hook) + analise de ablacao clima"
git push
rm -rf ccd_covariate_importance ccd_covariate_importance.zip
```

## Próxima run com GPU (pra completar os deep models)

```bash
python -m src.models.deep_panel <doenca> tft 42          # gera feature_importance_tft_<doenca>.csv
```
Repita para as 5 doenças. Depois me manda os 5 CSVs que eu consolido junto com a análise das
árvores num relatório único.
