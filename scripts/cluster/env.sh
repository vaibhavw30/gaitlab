# CLUSTER: environment for GaitLab on DeltaAI (sourced, not run; docs/CLUSTER.md).
#   source scripts/cluster/env.sh          login node: setup, downloads (internet allowed)
#   source scripts/cluster/env.sh job      inside a job: everything offline, nothing re-resolved
#
# ===== SET FROM ROUND 1 (2026-11-05; docs/CLUSTER.md "Storage") ===============================
# Big, file-heavy things (uv cache, venv, uv's Python, Hugging Face cache, Apptainer cache) go
# under GAITLAB_STORE, not $HOME: round 1 showed $HOME mostly full. NCSA recommends /work/nvme for
# venvs and asks for no symlinks from $HOME into /projects or /work, so these are env vars, not
# links. /work/nvme/bhhv exists (group delta_bhhv, setgid, group-writable) and is shared with the
# rest of the bhhv allocation.
GAITLAB_STORE="${GAITLAB_STORE:-/work/nvme/bhhv/vwudaru/gaitlab}"
# ==============================================================================================

if [[ -z $GAITLAB_STORE ]]; then
  echo "env.sh: set GAITLAB_STORE at the top of scripts/cluster/env.sh (round 1) first" >&2
  return 1 2>/dev/null || exit 1
fi
GAITLAB_ENV_MODE="${1:-login}"

export GAITLAB_REPO="$HOME/gaitlab"
export PATH="$HOME/.local/bin:$PATH" # uv
export UV_PYTHON=3.11
export UV_CACHE_DIR="$GAITLAB_STORE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$GAITLAB_STORE/uv-python"
export UV_PROJECT_ENVIRONMENT="$GAITLAB_STORE/venv" # same filesystem as the cache: hardlinks
export UV_FROZEN=1                                  # never re-lock on the cluster
export UV_NO_SYNC=1                                 # `uv run` never changes the venv
export HF_HOME="$GAITLAB_STORE/hf"                  # llm.runtime reads HF_HOME
export HF_HUB_DISABLE_XET=1
export APPTAINER_CACHEDIR="$GAITLAB_STORE/apptainer-cache"
export WANDB_MODE=disabled # W&B would turn on for CUDA runs; nodes have no internet
export PYTHONDONTWRITEBYTECODE=1

if [[ $GAITLAB_ENV_MODE == job ]]; then
  export UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
elif [[ $GAITLAB_ENV_MODE != login ]]; then
  echo "env.sh: mode must be 'login' or 'job', got '$GAITLAB_ENV_MODE'" >&2
  return 2 2>/dev/null || exit 2
fi
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$APPTAINER_CACHEDIR"
echo "env.sh ($GAITLAB_ENV_MODE): store=$GAITLAB_STORE venv=$UV_PROJECT_ENVIRONMENT HF_HOME=$HF_HOME"
