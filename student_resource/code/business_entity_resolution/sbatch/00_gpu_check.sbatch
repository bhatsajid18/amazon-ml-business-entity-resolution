#!/bin/bash
#SBATCH --job-name=ber_gpucheck
#SBATCH --partition=cse-gpu-all
#SBATCH --gres=gpu:P100-PCIE-16GB:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.log
#
# Always request the P100 explicitly: plain gpu:1 can land on an old-driver node.
# GPU smoke test + throughput benchmark. Confirms torch sees the P100 with sm_60
# kernels, then times LaBSE encoding and XLM-R pair scoring on real test records,
# and prints how long the full encode / cross-encoder passes would take.
# Submit from code/business_entity_resolution/:   sbatch sbatch/00_gpu_check.sbatch

source sbatch/env.sh
nvidia-smi
python src/gpu_check.py --data-dir "$DATA_DIR" --n 20000
