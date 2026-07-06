# ccd_final_package — pacote final: DeepAR + variância + READMEs

Resolve tudo que faltava de uma vez. Todos os arquivos foram gerados/testados com os dados
reais e as **mesmas funções do `make_results.py`** (nada calculado por fora ou de memória).

## O que este pacote resolve

| Item pendente | Status |
|---|---|
| **Resultados do DeepAR** | ✅ integrados — 5 CSVs brutos em `reports/` + métricas no `final_summary.csv`, `dtw_by_model.csv`, `ranking.csv`, `per_muni_*` |
| **Fix do DeepAR no código** | ✅ `src/models/deep_panel.py` com `train_deepar` corrigido (recurrent, sem clima) |
| **READMEs atualizados** | ✅ raiz, `notebooks/`, `results/` (regenerado com 12 modelos), `results/ensemble/` (marcado como desatualizado) |
| **Variância (desvio-padrão)** | ⚠️ ver nota abaixo — `mae_std_origin` preenchido; `mae_std_seed` fica NaN porque a run foi de 1 semente |

### Sobre a variância / `mae_std_seed`

O `final_summary.csv` tem duas colunas de desvio-padrão:
- **`mae_std_origin`** — desvio do MAE entre as 4 origens do rolling. **Preenchido para todos os
  modelos**, DeepAR incluído. É uma medida real de estabilidade temporal, e é a que dá pra
  reportar com o que foi rodado.
- **`mae_std_seed`** — desvio entre sementes de re-treino. **Fica NaN** porque a run (tanto a do
  PR #3 quanto a do DeepAR no Colab) usou **1 semente só (42)**. Isso não é um bug: pra preencher
  essa coluna é preciso rodar multi-seed (ex.: `python -m src.models.deep_panel <doenca> <modelo> 42,1,7`),
  o que multiplica o tempo de treino. **Decisão sua** se vale a pena pro artigo — o código já
  está pronto pra isso, é só rodar. Enquanto isso, reporte a variância via `mae_std_origin`.

## Impacto do DeepAR no resultado (verificado)

- Alinhamento rolling-origin conferido: DeepAR prevê as janelas certas (240/242/245/251 na 1ª
  origem, avançando por origem) — o fix funcionou.
- **Nenhum vencedor por doença muda** com o DeepAR: hanseníase→nhits, hepatite→lstm,
  hiv→catboost, sífilis→xgboost, tuberculose→seasonal_ma3.
- Mas o DeepAR **entra no top-3 em 4 pares** (doença × horizonte) e é **1º lugar em 2**
  (hiv_aids h=12 com MAE 2.21; tuberculose h=6 com MAE 2.51) — justifica tê-lo recuperado.

## Como aplicar (terminal)

Os arquivos em `repo_files/` espelham a estrutura do repo. Do diretório do teu clone:

```bash
cd ~/Documents/ccd_fin_clone/ccd_fin
git pull                                  # garante estar em dia com o GitHub

unzip -o ccd_final_package.zip            # extrai aqui
cp -r ccd_final_package/repo_files/* .    # sobrepõe os arquivos

git status                                # confira o que mudou (23 rastreados + reports/ novo)
git add -A
git commit -m "results: integra DeepAR (fix + metricas), atualiza READMEs e tabelas com 12 modelos"
git push

rm -rf ccd_final_package ccd_final_package.zip
```

> **Nota sobre `reports/`:** os 5 `deep_deepar_*.csv` vão em `reports/`, que normalmente é
> gitignored. O `git add -A` pode não pegá-los. Se você quer versioná-los (recomendado, pra
> reprodutibilidade), force com:
> ```bash
> git add -f reports/deep_deepar_*.csv
> ```
> Se preferir manter `reports/` fora do versionamento, pule isso — as métricas já estão todas
> consolidadas em `results/tables/`, que é o que importa pro artigo.

## Conteúdo do pacote

```
repo_files/
├── README.md                         # raiz, reescrito (12 modelos, fixes de validação)
├── notebooks/README.md               # corrigido (DEEP_MODELS/DEEP_EPOCHS reais)
├── src/models/deep_panel.py          # com fix do DeepAR
├── reports/deep_deepar_*.csv         # 5 CSVs brutos do DeepAR (Colab, seed 42)
└── results/
    ├── README.md                     # regenerado com DeepAR
    ├── ensemble/README.md            # marcado como desatualizado + como regenerar
    └── tables/
        ├── final_summary.csv         # 240 linhas (12 modelos × 5 × 4)
        ├── dtw_by_model.csv          # 60 linhas
        ├── ranking.csv               # top-3 por doença × horizonte, com DeepAR
        └── per_muni_*.csv            # todas com DeepAR incluído
```

## Ainda genuinamente pendente (fora do escopo deste pacote)

- **`mae_std_seed`** — precisa de run multi-seed (decisão sua, ver acima).
- **Importância de covariáveis** — precisa dos modelos treinados salvos (feature_importances_
  dos GBDT + interpret_output do TFT). Não dá pra extrair só das predições. Quando tiver os
  modelos salvos, me chama.
- **Ensemble** (`results/ensemble/`) — os números lá são de uma config antiga; regenera com
  `python notebooks/ensemble.py` quando tiver o `reports/` completo dos 12 modelos localmente.
