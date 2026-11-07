#!/usr/bin/env bash
# CLUSTER (login node only, needs internet): one-time setup, docs/CLUSTER.md step 3. No GPU work.
#   scripts/cluster/setup-login.sh uv       install uv into ~/.local/bin (dotfiles untouched)
#   scripts/cluster/setup-login.sh sync     venv from uv.lock: llm+dev, then llm+trt+dev
#   scripts/cluster/setup-login.sh models   download Gemma 3 1B and 270M into $HF_HOME
# The Hugging Face login is interactive and stays manual (CLUSTER.md step 3).
set -euo pipefail
cd "$HOME/gaitlab"
# shellcheck source=env.sh
source scripts/cluster/env.sh login

case "${1:-}" in
  uv)
    if ! command -v uv >/dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    fi
    uv --version
    ;;
  sync)
    # Step 1: what the LoRA work needs. Step 2 adds TensorRT on its own, so a TensorRT install
    # failure (its aarch64 wheels need glibc >= 2.35) leaves a working llm environment.
    uv sync --frozen --extra llm --extra dev
    if uv sync --frozen --extra llm --extra trt --extra dev; then
      echo "SYNC llm+trt+dev: OK"
    else
      echo "SYNC trt: FAILED; the llm+dev environment is kept. Paste the error above."
    fi
    uv run python -c "import platform, torch, transformers, peft, trl; print('python', platform.python_version(), '| torch', torch.__version__, 'built for CUDA', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__, '| trl', trl.__version__)"
    ;;
  models)
    for m in google/gemma-3-1b-it google/gemma-3-270m-it; do
      uv run hf download "$m"
    done
    ls "$HF_HOME/hub"
    du -sh "$HF_HOME" "$UV_PROJECT_ENVIRONMENT" "$UV_CACHE_DIR"
    ;;
  *)
    echo "usage: $0 uv|sync|models" >&2
    exit 2
    ;;
esac
