#!/bin/bash
#SBATCH --job-name=ber_finetune
#SBATCH --partition=cse-gpu-all
#SBATCH --gres=gpu:P100-PCIE-16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x_%j.log
#
# Round 2: fine-tune LaBSE on fold-A training pairs (misses of round 1 first, look-alike
# hard negatives) -> WORK_DIR/models/labse_ft. ~1 h on one P100 for 350k examples.
# If it runs out of GPU memory, resubmit with:  EXTRA="--batch 64" sbatch sbatch/08_finetune.sbatch
#   sbatch sbatch/08_finetune.sbatch

source sbatch/env.sh
# Round 3 example: BASE_MODEL=../../work/models/labse_ft FROM_TAG=labse1 OUT_MODEL=../../work/models/labse_ft2 \
#                  LR=1e-5 REVERSE_FRAC=0.5 MAX_EXAMPLES=450000 sbatch sbatch/08_finetune.sbatch
python src/finetune.py --from-tag "${FROM_TAG:-labse0}" --max-examples "${MAX_EXAMPLES:-350000}" \
  --base-model "${BASE_MODEL:-sentence-transformers/LaBSE}" --lr "${LR:-2e-5}" \
  --reverse-frac "${REVERSE_FRAC:-0.0}" ${OUT_MODEL:+--out "$OUT_MODEL"} ${EXTRA:-}
