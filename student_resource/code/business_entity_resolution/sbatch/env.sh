# Sourced by every sbatch job:   source sbatch/env.sh
# Always submit jobs from code/business_entity_resolution/ (Slurm starts the job in
# the directory you ran sbatch from, and the #SBATCH --output=logs/... paths are
# relative to it).
set -eo pipefail

CODE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "$CODE_DIR/sbatch/env.sh" ]]; then
  echo "ERROR: submit from code/business_entity_resolution/ (got $CODE_DIR)"; exit 1
fi
cd "$CODE_DIR"
ROOT="$(cd "$CODE_DIR/../.." && pwd)"

export CODE_DIR ROOT
export DATA_DIR="$ROOT/dataset"
export WORK_DIR="${WORK_DIR:-$ROOT/work}"        # embeddings, candidates, features, models
export OUT_DIR="$ROOT/output"                     # matching_results.tsv, candidate_pairs.tsv
# A TRANSFORMERS_CACHE left in ~/.bashrc overrides HF_HOME for model weights; drop it
# (and the other legacy cache vars) so every model file lives in one place.
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE SENTENCE_TRANSFORMERS_HOME HF_HUB_CACHE
export HF_HOME="$ROOT/hf_cache"                   # filled by setup_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1    # compute nodes have no internet
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$WORK_DIR" "$OUT_DIR" logs

# ---- conda env created by setup_env.sh
# shellcheck disable=SC1090
source "${CONDA_ACTIVATE:-/share/apps/anaconda3/2025.06/bin/activate}"
conda activate "${ENV_NAME:-ber}"

echo "== job ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-interactive}) on $(hostname) at $(date '+%F %T')"
echo "== python $(which python) | cpus ${SLURM_CPUS_PER_TASK:-?} | work dir $WORK_DIR"
if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
fi
