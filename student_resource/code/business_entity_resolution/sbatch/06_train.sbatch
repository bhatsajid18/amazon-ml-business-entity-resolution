#!/bin/bash
#SBATCH --job-name=ber_train
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
#
# LightGBM matcher + decision-rule tuning for macro F0.5 on the validation fold.
# Prints the validation F0.5, the best threshold and feature importances.
#   sbatch sbatch/06_train.sbatch

source sbatch/env.sh
# EXTRA passes extra options, e.g. leave-one-country-out (labels come from the data):
#   TAG=labse2 EXTRA="--train-countries India --val-countries US --out-suffix _loco_in2us" sbatch sbatch/06_train.sbatch
python src/train_match.py --tag "${TAG:-labse0}" ${EXTRA:-}
