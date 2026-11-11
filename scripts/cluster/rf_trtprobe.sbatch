#!/bin/bash
# CLUSTER: does a Torch-TensorRT engine execute on this node, and where does it hang? (docs/CLUSTER.md)
#   sbatch scripts/cluster/rf_trtprobe.sbatch                     gemma-bf16 gemma1l-bf16
#   sbatch scripts/cluster/rf_trtprobe.sbatch tiny gemma-bf16     only these (at most 3)
# trt_probe.py's own watcher reports a hung engine call (GPU state, gdb native stacks, Python
# stack). Each case also has a 10-minute ceiling; SIGABRT goes straight to python (not through
# uv), so faulthandler prints the stack. Prints diagnostics only; writes no result file.
#SBATCH --job-name=rf_trtprobe
#SBATCH --account=bhhv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100g
#SBATCH --time=00:35:00
#SBATCH --nice=0
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
cd "$HOME/gaitlab" || exit 1
source scripts/cluster/env.sh job || exit 1
export PYTHONFAULTHANDLER=1
ulimit -c 0 # an aborted case must not leave a multi-GB core file in ~/gaitlab
echo "== $(hostname) job ${SLURM_JOB_ID}"
cases=("$@")
((${#cases[@]})) || cases=(gemma-bf16 gemma1l-bf16)
if ((${#cases[@]} > 3)); then
  echo "at most 3 cases per job (3 x 10 min fits the 35-min limit)" >&2
  exit 2
fi
for c in "${cases[@]}"; do
  timeout --signal=ABRT 10m "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/cluster/trt_probe.py "$c"
  echo "PROBE $c EXIT $?"
done
