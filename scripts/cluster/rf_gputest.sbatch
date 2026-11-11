#!/bin/bash
# CLUSTER: step 4b, GPU smoke tests (docs/CLUSTER.md). Submit with: scripts/cluster/submit.sh rf_gputest
# Only after rf_gate printed "GATE RESULT PASS". Runs `pytest -m gpu` (TASKS M3 "GPU smoke test
# on the M3 host"). The tests write only to pytest's temp dirs, never to results/.
#SBATCH --job-name=rf_gputest
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100g
#SBATCH --time=00:45:00
#SBATCH --nice=0
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$HOME/gaitlab" || exit 1
source scripts/cluster/env.sh job || exit 1
# Each GPU test runs in its own child pytest (tests/conftest.py). A child still running after
# 20 min is killed with its stack printed, and the remaining tests run (the TensorRT test once
# hung until Slurm killed the whole job).
export GAITLAB_CHILD_TIMEOUT_S=1200 PYTHONFAULTHANDLER=1
ulimit -c 0
echo "== $(hostname) job ${SLURM_JOB_ID}"
uv run pytest -m gpu -v -rs -p no:cacheprovider
