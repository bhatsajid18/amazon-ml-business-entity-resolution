#!/bin/bash
#SBATCH --job-name=ber_eval
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.log
#
# Local score: writes a matching_results-style TSV for the held-out training fold C with
# the same model + decision rule as the test submission, then scores it against
# train_ground_truth.tsv with a stand-alone implementation of the macro F0.5 formula.
#   TAG=labse1 sbatch sbatch/09_evaluate.sbatch
# Output file: output/val_<TAG>_matching_results.tsv

source sbatch/env.sh
python src/evaluate.py --tag "${TAG:-labse1}"
