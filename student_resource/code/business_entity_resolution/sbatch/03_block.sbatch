#!/bin/bash
#SBATCH --job-name=ber_block
#SBATCH --partition=cse-gpu-all
#SBATCH --gres=gpu:P100-PCIE-16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.log
#
# Blocking: exact bidirectional top-k search per (split, country, source).
# 10 tasks = train {India, US} x {S2, S3} + test {France, India, US} x {S2, S3}.
# Run after ALL encode tasks of the same TAG have finished. Train tasks also print the
# recall ceiling for several (k_fwd, k_rev) cut-offs.
#
#   sbatch sbatch/03_block.sbatch                   # round 1 (TAG=labse0)
#   TAG=labse1 sbatch sbatch/03_block.sbatch        # round 2
# Check the task list first:  python src/block.py --count   (must be 10 for --array=0-9)

source sbatch/env.sh
# Optional: OUT_TAG=<name> writes candidates under a different tag; EXCLUDE=0-19 leaves 20% of
# train S1 entities out of the index so train has test-like density of unmatched S2/S3 records.
python src/block.py --task "$SLURM_ARRAY_TASK_ID" --tag "${TAG:-labse0}" \
  ${OUT_TAG:+--out-tag "$OUT_TAG"} ${EXCLUDE:+--exclude-s1-buckets "$EXCLUDE"}
