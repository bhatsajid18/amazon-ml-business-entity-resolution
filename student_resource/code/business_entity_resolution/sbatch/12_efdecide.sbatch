#!/bin/bash
#SBATCH --job-name=ber_efdec
#SBATCH --partition=cse-cpu-all
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.log
#
# Per-record expected-F0.5 decision rule (and optional averaging of runs that share one
# candidate set), tuned on validation fold C; writes the test submission only if it beats the
# reference run's threshold rule by --min-gain (default 0.0002).
#   TAGS=labse2k OUT_TAG=labse2k_ef sbatch sbatch/12_efdecide.sbatch
#   TAGS=labse2k,labse2kb OUT_TAG=labse2k_ens sbatch sbatch/12_efdecide.sbatch
# Output: output/history/<OUT_TAG>_<time>/ (+ output/matching_results.tsv, candidate_pairs.tsv)

source sbatch/env.sh
python src/efdecide.py --tags "${TAGS:?set TAGS}" --out-tag "${OUT_TAG:-ef}" ${EXTRA:-}
