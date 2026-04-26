# ccd_fin

Predição mensal de 5 doenças (hanseníase, hepatite, HIV/AIDS, sífilis, tuberculose) em 23 municípios de São Paulo, usando PyTorch + GPU.

## O que tem aqui

- **`src/data/`** — pipeline de dados (wide→long, painel mensal, EDA)
- **`src/models/`** — baselines (SARIMA, LightGBM) + deep learning (TFT, N-HiTS via `pytorch-forecasting`)
- **`src/eval/`** — métricas (MAE, RMSE, sMAPE, WQL)
- **`notebooks/train_all.ipynb`** — pipeline completo end-to-end com diagnósticos visuais

## Setup

```bash
conda create -n ccd python=3.11 -y
conda activate ccd
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pandas numpy pyarrow openpyxl odfpy python-calamine scikit-learn matplotlib seaborn \
            lightning tensorboard pytorch-forecasting statsforecast lightgbm optuna pyyaml \
            jupyter nbformat ipykernel
python -m ipykernel install --user --name ccd --display-name "Python (ccd)"
```

## Dados

Os arquivos brutos (`*.xlsx`, `*.ods`, `*.csv`) **não estão no repositório** — coloque-os na raiz e em `Taxas/` antes de rodar.

Estrutura esperada:
- Doenças (mensal 2000-2023): `<doenca>_00_23.xlsx` (contagens) + `Taxas/TX_<doenca>_00_23.xlsx` (taxas)
- Clima (mensal 1999-2023): `Evapot_SP.xlsx`, `Precip_SP.xlsx`, `Temp_Min_SP.xlsx`, `Temp_Max_SP.ods`, `Umid_SP.xlsx`
- Socioeconômico (anual 2000-2023): `Pop_Geral_SP.xlsx`, `Indice_PPC_SP.xlsx`, `Urban_SP.xlsx`, `Dens_demog_SP.csv`

## Como rodar

1. Coloque os dados na raiz (ver acima).
2. Abra `notebooks/train_all.ipynb` no VS Code.
3. Selecione kernel **Python (ccd)**.
4. Run All.

Treina baselines + 2 modelos deep (TFT, N-HiTS) por 4 origens × 5 doenças, com early stopping. Gera figuras em `reports/figures/` e tabelas em `reports/`.

## Estrutura

```
src/
├── data/      # load.py, build_panel.py, eda.py
├── models/    # baselines.py, deep_panel.py
├── eval/      # metrics.py
└── utils/     # paths.py, splits.py
notebooks/     # train_all.ipynb (+ build_notebook.py para regenerar)
```
