#!/bin/bash
# run_experiments.sh
# Reproduz o pipeline ccd_fin de ponta a ponta e (re)gera as tabelas de
# resultados (MAE/RMSE/R²/SMAPE por doença e por município) em results/.
#
# Uso:
#   bash run_experiments.sh
#
# Pré-requisitos:
#   - ambiente com as deps do README (conda activate ccd) + GPU recomendada
#   - dados brutos (NÃO versionados) posicionados:
#       raiz/  -> <doenca>_00_23.xlsx, Evapot_SP.xlsx, Precip_SP.xlsx,
#                 Temp_Min_SP.xlsx, Temp_Max_SP.ods, Umid_SP.xlsx,
#                 Pop_Geral_SP.xlsx, Indice_PPC_SP.xlsx, Urban_SP.xlsx,
#                 Dens_demog_SP.csv
#       Taxas/ -> TX_<doenca>_00_23.xlsx
set -euo pipefail

# Sempre roda a partir da raiz do repo (onde está este script)
cd "$(dirname "$0")"

echo "============================================================"
echo "    Reproducing ccd_fin Experiments"
echo "============================================================"

# 0. Python: usa $PYTHON se definido, senão python3
PY="${PYTHON:-python3}"
echo "=> Python: $($PY --version 2>&1)"

# 1. Checagem dos dados brutos (raiz + Taxas/)
missing=0
for f in hanseniase_00_23.xlsx hepatite_00_23.xlsx hiv_aids_00_23.xlsx \
         sifilis_00_23.xlsx tuberculose_00_23.xlsx \
         Evapot_SP.xlsx Precip_SP.xlsx Temp_Min_SP.xlsx Temp_Max_SP.ods \
         Umid_SP.xlsx Pop_Geral_SP.xlsx Indice_PPC_SP.xlsx Urban_SP.xlsx \
         Dens_demog_SP.csv; do
    [ -f "$f" ] || { echo "   FALTA (raiz): $f"; missing=1; }
done
for d in hanseniase hepatite hiv_aids sifilis tuberculose; do
    [ -f "Taxas/TX_${d}_00_23.xlsx" ] || { echo "   FALTA (Taxas/): TX_${d}_00_23.xlsx"; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo "ERRO: arquivos de dados ausentes (ver acima). Posicione-os e rode de novo."
    exit 1
fi
echo "=> Dados brutos: OK"

# 2. Descobre o diretório de cache resolvido pelo paths.py (portável)
CACHE_DIR="$($PY -c 'from src.utils.paths import CACHE; print(CACHE)')"
echo "=> Cache dir: $CACHE_DIR"

# 3. Limpa relatórios e o parquet do painel para forçar re-execução completa.
#    (Para reaproveitar deep models já treinados, comente as duas linhas abaixo.)
echo "=> Limpando reports/ e cache do painel..."
mkdir -p reports
rm -f reports/*.csv
rm -f "${CACHE_DIR}/processed/panel_23munis.parquet" 2>/dev/null || true

# 4. Pipeline de treino (constrói painel + baselines + deep models)
echo "=> 1/3: train_all.py (pode levar ~1h+ em GPU)..."
$PY notebooks/train_all.py

# 5. Ensemble
echo "=> 2/3: ensemble.py..."
$PY notebooks/ensemble.py

# 6. Consolida tabelas e figuras em results/
echo "=> 3/3: make_results.py (tabelas MAE/RMSE/R²/SMAPE por doença e município)..."
$PY scripts/make_results.py

echo "============================================================"
echo "    Pipeline concluído!"
echo "    Tabelas em results/tables/  | figuras em results/figures/"
echo "============================================================"
