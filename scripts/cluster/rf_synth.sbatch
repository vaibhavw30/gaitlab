#!/bin/bash
# CLUSTER: step 5, synthetic 1B LoRA dry run (docs/CLUSTER.md). Submit with: scripts/cluster/submit.sh rf_synth
# Synthetic SFT only (--allow-synthetic): plumbing, never reported (CLAUDE.md rule 3).
#SBATCH --job-name=rf_synth
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100g
#SBATCH --time=00:45:00
#SBATCH --nice=1000
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$HOME/gaitlab"
source scripts/cluster/env.sh job
echo "== $(hostname) job ${SLURM_JOB_ID}"
uv run gaitlab train --model 1b --sft-dir data/synthetic/sft --allow-synthetic --max-steps 20 --device cuda
uv run gaitlab predict --synthetic --model 1b --device cuda
ls -d runs/1b-*
