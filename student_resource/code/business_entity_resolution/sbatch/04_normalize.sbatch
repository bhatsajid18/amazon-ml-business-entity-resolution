#!/bin/bash
#SBATCH --job-name=ber_norm
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.log
#
# Text normalization for string features (romanize Indic scripts, expand abbreviations,
# extract numbers) -> WORK_DIR/norm/<split>_<source>.parquet. Task 0 = train, 1 = test.
# Needs no embeddings, so it can run while the GPU encoding is in progress.
#   sbatch sbatch/04_normalize.sbatch

source sbatch/env.sh
SPLITS=(train test)
python src/normalize.py --split "${SPLITS[$SLURM_ARRAY_TASK_ID]}"
