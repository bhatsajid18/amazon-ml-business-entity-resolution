#!/bin/bash
#SBATCH --job-name=ber_keyblk
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.log
#
# Second candidate channel (exact rare keys: name words, postal+house number, house number +
# street word) merged into the embedding candidates. Same 10 shards as 03_block.sbatch;
# train shards print the recall ceiling before/after. Needs work/norm/v2 (04_normalize).
#   TAG=labse2 OUT_TAG=labse2k EXCLUDE=0-19 sbatch sbatch/11_keyblock.sbatch

source sbatch/env.sh
python src/key_block.py --task "$SLURM_ARRAY_TASK_ID" --tag "${TAG:-labse2}" --out-tag "${OUT_TAG:-labse2k}" \
  ${EXCLUDE:+--exclude-s1-buckets "$EXCLUDE"}
