# Pacote final — tudo que falta commitar

Estado do repo conferido agora: o último commit é `305064c` (importância de covariáveis).
**Nada dos três pacotes anteriores foi aplicado** — este zip consolida tudo num só, para um
único commit (ou dois, se preferir separar as figuras).

---

## O que este pacote contém

### Documentação nova (2 arquivos)
- **`docs/ARQUITETURAS.md`** — referência para a seção de métodos: protocolo experimental,
  janelas de entrada (encoder 36) e saída (12), batch 128, LR 1e-3, épocas (teto 200 +
  EarlyStopping patience 7), topologia exata dos 5 deep models separando o que o projeto fixa
  do que é default do `pytorch-forecasting 1.8.0`, covariáveis por modelo, hiperparâmetros das
  árvores e estatísticos, e o registro explícito de que **não houve tuning sistemático**.
- **`docs/RESSALVA_ZERO_INFLACAO.md`** — a limitação mais importante achada: o ranking por MAE
  é enganoso nas doenças zero-infladas. Em hanseníase (88% de zeros) o `nhits` vence em MAE
  mas prevê praticamente zero sempre (`y_pred` mín. 8,4×10⁻²², média 0,051) e detecta só
  **18,5%** dos eventos reais, enquanto o `xgboost` tem MAE 17% pior e detecta **3× mais**.
  Traz as três formas de tratar isso no artigo.

### Código (1 arquivo)
- **`notebooks/forecast_plots.py`** — corrigidas as listas hardcoded (`DEEP_MODELS` tinha 3
  modelos, `BASELINE_MODELS` tinha 5) e a `PALETTE`, que só tinha cor para 8 modelos enquanto
  a legenda usa `PALETTE[m]` com indexação direta — daria `KeyError` nos 4 novos. Era o mesmo
  bug já corrigido em `train_all`/`ensemble`/`make_results`, que tinha passado batido aqui.

### Dados brutos (28 CSVs em `reports/`)
O `reports/` completo dos 12 modelos, validado: `time_idx` avança por origem
(240→252→264→276) e a checagem de sanidade baseline×deep passa **100%** (1840 linhas, 0
divergências). Versionar isso torna as figuras e tabelas reproduzíveis sem depender da
máquina remota.

### Tabelas regeneradas
`final_summary.csv` (240 linhas, 12 modelos), `dtw_by_model.csv`, `ranking.csv` e todas as
`per_muni_*`. Regeneradas **do zero pelo pipeline**, não por integração manual. Validação
cruzada: os vencedores por doença bateram exatamente com o resultado anterior.

### Ensemble re-rodado
`results/ensemble/` inteiro — o README que estava marcado como desatualizado agora tem dados
reais dos 12 modelos. Resultado novo: o `all_mean` (MAE 1,717) passou a bater as estratégias
top-3 (1,749), e o ensemble tem MAE médio melhor que a média dos melhores individuais (1,803)
**mas perde em 20/20 pares** doença×horizonte. Vale reportar essa nuance no texto.

### Figuras (202 arquivos)
- `results/figures/forecasts/` — 25 figuras, agora **com o DeepAR** (5 novas)
- `results/forecasts_detailed/` — 115 figuras (23 municípios × 5 doenças) **com os 12 modelos**
- `results/forecasts_grid/` — 60 grades modelo×doença
- `results/figures/comparison/` — heatmaps atualizados + **`mae_vs_recall_tradeoff.png`** (nova,
  torna visível a armadilha da zero-inflação)
- `results/ensemble/figures/` — regeneradas

### READMEs atualizados
- `README.md` — link para `docs/ARQUITETURAS.md` na §5 e roadmap corrigido (a importância nas
  árvores já está feita)
- `results/README.md` — regenerado pelo pipeline + aviso no topo apontando para a ressalva da
  zero-inflação, para ninguém citar os vencedores sem lê-la

---

## Instruções

```bash
cd ~/Documents/ccd_fin_clone/ccd_fin
git pull                                  # garante estar em 305064c

unzip -o ccd_final.zip
cp -r ccd_final/repo_files/* .

git status                                # confira: ~242 arquivos alterados
git add -A
git add -f reports/*.csv                  # reports/ é gitignored — o -f é obrigatório

git commit -m "docs: arquiteturas e ressalva de zero-inflacao; results: 12 modelos em tabelas, figuras e ensemble; fix: forecast_plots"
git push

rm -rf ccd_final ccd_final.zip
git status                                # deve ficar "working tree clean"
```

### Se preferir dois commits (recomendado se o push de 100 MB ficar lento)

```bash
# 1) código, docs, tabelas e dados
git add README.md docs/ notebooks/forecast_plots.py results/README.md results/tables/ results/ensemble/
git add -f reports/*.csv
git commit -m "docs: arquiteturas e ressalva de zero-inflacao; results: tabelas e ensemble com 12 modelos"
git push

# 2) figuras
git add -A
git commit -m "figs: forecasts com DeepAR, detalhadas e grades com 12 modelos, MAE vs recall"
git push
```

> **Alternativa sem subir 100 MB de PNG:** pule as figuras deste pacote e, depois de commitar o
> resto (o `reports/` fica versionado), regenere localmente:
> ```bash
> python scripts/make_results.py
> python notebooks/forecast_plots.py
> ```

---

## Depois deste commit, o balanço dos itens da revisão

| Item | Status |
|---|---|
| ARIMA/Prophet/LSTM/GRU/XGBoost no benchmark | ✅ 12 modelos |
| DTW | ✅ |
| Recall de eventos não-zero | ✅ (e revelou a ressalva da zero-inflação) |
| Vazamentos de validação | ✅ 4 corrigidos, documentados e validados |
| Importância de covariáveis | ⚠️ árvores + ablação prontas; **deep models faltam rodar em GPU** |
| Variância (desvio-padrão) | ⚠️ `mae_std_origin` pronto; **`mae_std_seed` precisa de run multi-seed** |
| Detalhes de arquitetura | ✅ `docs/ARQUITETURAS.md` |
| Figuras atualizadas | ✅ |

Os dois pendentes dependem só de tempo de GPU, nenhum precisa de código novo:
```bash
python -m src.models.deep_panel <doenca> tft 42          # importância do TFT
python -m src.models.deep_panel <doenca> <modelo> 42,1,7  # variância entre sementes
```
