#!/bin/bash
#SBATCH --job-name=ber_predict
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
#
# Score test candidates, apply the tuned decision rule, write output/matching_results.tsv
# and output/candidate_pairs.tsv (+ a copy under output/history/), then run the validator.
#   sbatch sbatch/07_predict.sbatch

source sbatch/env.sh
python src/predict.py --tag "${TAG:-labse0}"
