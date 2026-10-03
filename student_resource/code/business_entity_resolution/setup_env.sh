#!/usr/bin/env bash
# One-time environment setup for the Business Entity Resolution pipeline.
#
# Run ONCE on the LOGIN node (needs internet; GPU compute nodes have none):
#     cd ~/ber/student_resource/code/business_entity_resolution
#     bash setup_env.sh
#
# Creates a NEW conda env "ber" (your edl_torch env is not touched) with the same
# Python / PyTorch combination already proven on the cse-gpu-all P100 nodes:
# Python 3.11 + torch 2.4.1 built for CUDA 12.1. Then installs requirements.txt,
# pre-downloads LaBSE and XLM-R into student_resource/hf_cache, creates the
# logs/work/output folders and writes requirements.lock.txt.
set -eo pipefail

CONDA_ACTIVATE="${CONDA_ACTIVATE:-/share/apps/anaconda3/2025.06/bin/activate}"
ENV_NAME="${ENV_NAME:-ber}"

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../code/business_entity_resolution
ROOT="$(cd "$CODE_DIR/../.." && pwd)"                        # .../student_resource

if [[ ! -d "$ROOT/dataset/train" ]]; then
  echo "ERROR: expected $ROOT/dataset/train."
  echo "Unzip ber_setup.zip INSIDE student_resource/ so this file is at"
  echo "student_resource/code/business_entity_resolution/setup_env.sh"
  exit 1
fi
echo "== code dir: $CODE_DIR"
echo "== root dir: $ROOT"

# ---------------------------------------------------------------- 1. conda env
# shellcheck disable=SC1090
source "$CONDA_ACTIVATE"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "== env $ENV_NAME exists, reusing it"
else
  # conda-forge only: sidesteps the Anaconda "defaults" channel terms-of-service prompt.
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.11 pip
fi
conda activate "$ENV_NAME"
echo "== python: $(which python)  ($(python --version))"

# ---------------------------------------------------------------- 2. packages
export PIP_NO_CACHE_DIR=1          # keeps several GB of wheels out of your home quota
python -m pip install --upgrade pip
pip install "torch==2.4.1+cu121" --index-url https://download.pytorch.org/whl/cu121
pip install -r "$CODE_DIR/requirements.txt"

# The login node has no GPU, so torch.cuda.get_arch_list() is empty here by design;
# the sm_60 kernel check runs on a real P100 in sbatch/00_gpu_check.sbatch.
python -c "import torch; print('== torch', torch.__version__, '| CUDA', torch.version.cuda)"

# ---------------------------------------------------------------- 3. models
# Compute nodes are offline, so cache the weights now; jobs run with HF_HUB_OFFLINE=1.
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE SENTENCE_TRANSFORMERS_HOME HF_HUB_CACHE
export HF_HOME="$ROOT/hf_cache"
mkdir -p "$HF_HOME"
python - <<'EOF'
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
SentenceTransformer("sentence-transformers/LaBSE")
print("== cached sentence-transformers/LaBSE")
AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
AutoModel.from_pretrained("FacebookAI/xlm-roberta-base")
print("== cached FacebookAI/xlm-roberta-base")
EOF

# ---------------------------------------------------------------- 4. dirs + lock
mkdir -p "$CODE_DIR/logs" "$ROOT/output" "$ROOT/work"/{emb,cand,feat,models}
pip freeze > "$CODE_DIR/requirements.lock.txt"

echo
echo "== DONE. Env '$ENV_NAME' ready; exact versions in requirements.lock.txt"
echo "   Next, from $CODE_DIR:"
echo "     sbatch sbatch/00_gpu_check.sbatch"
echo "     sbatch sbatch/01_data_stats.sbatch"
