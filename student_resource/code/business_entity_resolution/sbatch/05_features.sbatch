#!/bin/bash
#SBATCH --job-name=ber_feat
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.log
#
# Pair features for every candidate pair after the k_fwd / k_rev cut-off.
# Task 0 = train (needs train blocking done), 1 = test (needs test blocking done).
#   sbatch --array=0 sbatch/05_features.sbatch             # train only
#   K_FWD=10 K_REV=3 sbatch --array=1 sbatch/05_features.sbatch
#   Round 2 (encoder fine-tuned on fold A, so LightGBM must train on fold B only):
#   TAG=labse1 TRAIN_FOLDS=B TRAIN_FRAC=0.5 sbatch sbatch/05_features.sbatch
#   Rarity-weighted name features:  EXTRA=--idf
# Output: WORK_DIR/feat/<TAG>/<split>.parquet

source sbatch/env.sh
SPLITS=(train test)
python src/features.py --split "${SPLITS[$SLURM_ARRAY_TASK_ID]}" --tag "${TAG:-labse0}" \
  --k-fwd "${K_FWD:-10}" --k-rev "${K_REV:-3}" \
  --train-folds "${TRAIN_FOLDS:-AB}" --train-frac "${TRAIN_FRAC:-0.25}" ${EXTRA:-}
