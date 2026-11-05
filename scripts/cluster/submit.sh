#!/usr/bin/env bash
# CLUSTER (login node): budget check, then sbatch one rf_* job (docs/CLUSTER.md steps 4-6).
#   scripts/cluster/submit.sh rf_gate | rf_gputest | rf_synth
# Refuses jobs that don't ask for exactly 1 GPU with --nice=1000 and an rf_ name, and anything
# that could take the tally past 4.0 GH200-hours (budget.sh exit 3).
set -euo pipefail
cd "$HOME/gaitlab"
job="${1:?usage: $0 rf_gate|rf_gputest|rf_synth}"
file="scripts/cluster/$job.sbatch"
[[ -f $file ]] || { echo "no $file" >&2; exit 2; }
for line in "--job-name=$job" "--gpus-per-node=1" "--nice=1000" "--account=bhhv-dtai-gh"; do
  grep -qx -- "#SBATCH $line" "$file" || { echo "refusing: $file lacks '#SBATCH $line'" >&2; exit 2; }
done
[[ $job == rf_* ]] || { echo "refusing: job names start with rf_" >&2; exit 2; }
limit="$(sed -n 's/^#SBATCH --time=//p' "$file")"
part="$(sed -n 's/^#SBATCH --partition=//p' "$file")"
scripts/cluster/budget.sh --next "$limit" "$part" || { echo "NOT SUBMITTED" >&2; exit 3; }
mkdir -p logs
sbatch "$file"
squeue --me
