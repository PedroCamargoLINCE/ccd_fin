#!/bin/bash
# run_experiments.sh
# Script to run the complete ccd_fin pipeline and reproduce the results tables.
set -e

echo "============================================================"
echo "    Reproducing ccd_fin Experiments"
echo "============================================================"

# 1. Environment Activation
# (Assuming the partner has already run the setup commands from README.md)
# conda activate ccd

# 2. Check for Raw Data
if [ ! -d "Taxas" ] || [ -z "$(ls -A ./*.xlsx 2>/dev/null)" ]; then
    echo "WARNING: Raw data files (*.xlsx, Taxas/*.xlsx) seem to be missing from the root directory."
    echo "Please ensure the non-versioned datasets are placed in the root folder before proceeding."
    sleep 3
fi

# 3. Clear existing caches to force a full re-run
echo "=> Cleaning up old reports and caches..."
rm -rf reports/*.csv
rm -rf /C/temp/ccd_cache/processed/panel_23munis.parquet 2>/dev/null || true
rm -rf C:/temp/ccd_cache/processed/panel_23munis.parquet 2>/dev/null || true

# 4. Run the main training pipeline
echo "=> 1/3: Running train_all.py (This will take ~1h+ on a GPU)..."
python notebooks/train_all.py

# 5. Run the ensemble evaluations
echo "=> 2/3: Running ensemble.py..."
python notebooks/ensemble.py

# 6. Re-generate the results folder (markdowns and missing CSV tables)
echo "=> 3/3: Generating final results tables and metrics..."
python scripts/make_results.py

echo "============================================================"
echo "    Pipeline completed successfully!"
echo "    The generated tables are now available inside results/"
echo "============================================================"
