#!/bin/bash
# CLUSTER: step 4a, the gate (docs/CLUSTER.md). Submit with: scripts/cluster/submit.sh rf_gate
# Confirms driver/CUDA 13, torch CUDA build on a GH200, and the TensorRT imports. Nothing else
# runs on a GPU until this passes.
#SBATCH --job-name=rf_gate
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100g
#SBATCH --time=00:15:00
#SBATCH --nice=0
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$HOME/gaitlab" || exit 1
source scripts/cluster/env.sh job || exit 1
echo "== $(hostname) job ${SLURM_JOB_ID}"
grep PRETTY_NAME /etc/os-release
nvidia-smi
uv run python scripts/cluster/gate.py
