#!/usr/bin/env bash
# CLUSTER (login node): budget check, then sbatch one rf_* job (docs/CLUSTER.md steps 4-6).
#   scripts/cluster/submit.sh rf_gate | rf_gputest | rf_synth
# Refuses jobs that don't ask for exactly 1 GPU with --nice=0 and an rf_ name. If the job could
# take the tally past 14.0 of the 15.0 GH200-hours (budget.sh exit 3), it asks before submitting.
set -euo pipefail
cd "$HOME/gaitlab"
job="${1:?usage: $0 rf_gate|rf_gputest|rf_synth}"
file="scripts/cluster/$job.sbatch"
[[ -f $file ]] || { echo "no $file" >&2; exit 2; }
for line in "--job-name=$job" "--gpus-per-node=1" "--nice=0" "--account=bhhv-dtai-gh"; do
  grep -qx -- "#SBATCH $line" "$file" || { echo "refusing: $file lacks '#SBATCH $line'" >&2; exit 2; }
done
[[ $job == rf_* ]] || { echo "refusing: job names start with rf_" >&2; exit 2; }
limit="$(sed -n 's/^#SBATCH --time=//p' "$file")"
part="$(sed -n 's/^#SBATCH --partition=//p' "$file")"
if ! scripts/cluster/budget.sh --next "$limit" "$part"; then
  read -r -p "Past 14.0 of 15.0 GH200-hours. Submit $job anyway? [y/N] " ok </dev/tty || ok=""
  [[ $ok == [yY] ]] || { echo "NOT SUBMITTED" >&2; exit 3; }
fi
mkdir -p logs
sbatch "$file"
squeue --me
