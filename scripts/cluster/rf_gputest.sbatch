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
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:20:00
#SBATCH --nice=1000
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$HOME/gaitlab" || exit 1
source scripts/cluster/env.sh job || exit 1
echo "== $(hostname) job ${SLURM_JOB_ID}"
uv run pytest -m gpu -v -rs -p no:cacheprovider
