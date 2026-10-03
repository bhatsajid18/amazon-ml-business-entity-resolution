#!/bin/bash
#SBATCH --job-name=ber_diag
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.log
#
# Test-set diagnostics per country label (needs work/pred/<TAG>_test.parquet from 07_predict):
#   TAG=labse2 SHOW=France sbatch sbatch/10_diagnose.sbatch
# Probe file with one country's rows emptied (upload it next to the full file):
#   RUN=labse2_20260927_051906 BLANK=France sbatch sbatch/10_diagnose.sbatch

source sbatch/env.sh
if [[ -n "${BLANK:-}" ]]; then
  python src/diagnose.py blank --run "$RUN" --country "$BLANK"
else
  python src/diagnose.py stats --tag "${TAG:-labse2}" ${SHOW:+--show "$SHOW"} --n "${N:-30}"
fi
