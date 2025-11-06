#!/usr/bin/env bash
set -euo pipefail

# Example run: assumes winequalityN.csv is in current directory
DATAFILE="winequalityN.csv"
OUTDIR="reports"
MODELDIR="models"

python decisiontree_generic.py --data "${DATAFILE}" --output-dir "${OUTDIR}" --models-dir "${MODELDIR}" --test-size 0.2 --quality-threshold 6 --node-limit 25 --n-jobs -1
echo "Run complete. Check ${OUTDIR} and ${MODELDIR}."