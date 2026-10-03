#!/bin/bash
#SBATCH --job-name=ber_stats
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
#
# Dataset profiling (CPU only). Its answers fix the pipeline's switches:
# forward top-k, one-parent rule, per-country index shards, max sequence length.
# Submit from code/business_entity_resolution/:   sbatch sbatch/01_data_stats.sbatch

source sbatch/env.sh
python src/data_stats.py --data-dir "$DATA_DIR" --out "$WORK_DIR/data_stats.json"
