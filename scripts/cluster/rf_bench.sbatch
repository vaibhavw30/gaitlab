#!/bin/bash
# CLUSTER: B2 TensorRT bench dry run on the synthetic test prompts (docs/CLUSTER.md).
#   sbatch scripts/cluster/rf_bench.sbatch
# Synthetic: plumbing only, never reported (CLAUDE.md rule 3). Needs a synthetic 1B run in runs/
# (rf_synth) and data/synthetic/sft/test.jsonl. Writes results/synthetic/bench/trt.json.
# SIGABRT goes straight to python (not through uv), so a hang prints its stack before the limit.
#SBATCH --job-name=rf_bench
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
export PYTHONFAULTHANDLER=1
ulimit -c 0 # an aborted run must not leave a multi-GB core file in ~/gaitlab
echo "== $(hostname) job ${SLURM_JOB_ID}"
git status --porcelain --untracked-files=no | sed 's/^/DIRTY /'
timeout --signal=ABRT 40m "$UV_PROJECT_ENVIRONMENT/bin/gaitlab" bench --trt --synthetic --model 1b
echo "BENCH EXIT $?"
