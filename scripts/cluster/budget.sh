#!/usr/bin/env bash
# CLUSTER (login node): charged GH200-hours of rf_* jobs since 2026-11-04 against the 15.0 approved.
#   scripts/cluster/budget.sh                              the tally
#   scripts/cluster/budget.sh --next 00:20:00 [partition]  plus the next job's worst case
# The 15.0 is the budget, not a hard cap. Exit 3 (and an ASK line) if the tally is at or past 14.0,
# or the next job could take it past 14.0; submit.sh then asks before submitting.
#
# Charging (docs/CLUSTER.md "Charging"): each job is billed for the number of GH200s needed for
# its GPUs, CPU cores (72 per GH200) or memory (110 GB per GH200), whichever is largest, times the
# partition's charge factor (ghx4 1.0, ghx4-interactive 2.0), times elapsed wall time.
set -euo pipefail
START=2026-11-04
CAP=15.0
STOP=14.0
nxt=""
part=ghx4
if [[ ${1:-} == --next ]]; then
  nxt="${2:?usage: $0 --next HH:MM:SS [partition]}"
  part="${3:-ghx4}"
fi

sacct -u "$USER" -S "$START" -X -n -P --format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES |
  awk -F'|' -v cap="$CAP" -v stop="$STOP" -v nxt="$nxt" -v npart="$part" '
function ceil(x) { return (x == int(x)) ? x : int(x) + 1 }
function factor(p) { return (p ~ /interactive/) ? 2.0 : 1.0 }
function gib(s,    n, u) {
  n = s + 0; u = substr(s, length(s))
  if (u == "T") return n * 1024; if (u == "M") return n / 1024; if (u == "K") return n / 1048576
  return n
}
$2 ~ /^rf_/ {
  g = 0; c = 0; m = 0; b = "-"
  k = split($6, tres, ",")
  for (i = 1; i <= k; i++) {
    split(tres[i], kv, "=")
    if (kv[1] == "gres/gpu") g = kv[2]; else if (kv[1] == "cpu") c = kv[2]
    else if (kv[1] == "mem") m = gib(kv[2]); else if (kv[1] == "billing") b = kv[2]
  }
  eq = g; if (ceil(c / 72) > eq) eq = ceil(c / 72); if (ceil(m / 110) > eq) eq = ceil(m / 110)
  h = $5 / 3600 * eq * factor($3); used += h; raw += $5 / 3600 * g
  printf "%-10s %-11s %-17s %-10s gpus=%s cpus=%s mem=%.0fG billing=%s charged=%.3f\n", \
    $1, $2, $3, $4, g, c, m, b, h
}
END {
  printf "USED %.2f / %.1f GH200-hours (raw GPU-hours %.2f), REMAINING %.2f\n", used, cap, raw, cap - used
  rc = 0
  if (used >= stop) { print "ASK: at or past " stop " GH200-hours; confirm before starting any job"; rc = 3 }
  if (nxt != "") {
    n = split(nxt, t, ":")
    secs = (n == 3) ? t[1] * 3600 + t[2] * 60 + t[3] : t[1] * 60 + t[2]
    w = secs / 3600 * factor(npart)
    printf "NEXT worst case %.2f GH200-hours (%s x 1 GPU x factor %.1f on %s); tally would be %.2f\n", \
      w, nxt, factor(npart), npart, used + w
    if (used + w > stop) { print "ASK: the next job could pass " stop " GH200-hours; confirm first"; rc = 3 }
  }
  exit rc
}'
