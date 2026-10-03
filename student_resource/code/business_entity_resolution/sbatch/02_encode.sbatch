#!/bin/bash
#SBATCH --job-name=ber_encode
#SBATCH --partition=cse-gpu-all
#SBATCH --gres=gpu:P100-PCIE-16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-27%8
#SBATCH --output=logs/%x_%A_%a.log
#
# Encode every record with LaBSE: 28 tasks of up to 1M rows each (train S1/S2/S3 =
# 3/6/6 chunks, test S1/S2/S3 = 2/5/6). Each task takes ~20 min on a P100; Slurm runs
# up to 8 at once (or fewer if your GPU limit is lower). Finished chunks are skipped,
# so resubmitting the same command only redoes missing ones.
#
# Round 1 (off-the-shelf LaBSE):   sbatch sbatch/02_encode.sbatch
# Round 2 (fine-tuned model):      TAG=labse1 MODEL=../../work/models/labse_ft sbatch sbatch/02_encode.sbatch
# Check the task count first:       python src/encode.py --count   (must be 28 for --array=0-27)

source sbatch/env.sh
python src/encode.py --task "$SLURM_ARRAY_TASK_ID" \
  --tag "${TAG:-labse0}" --model "${MODEL:-sentence-transformers/LaBSE}"
