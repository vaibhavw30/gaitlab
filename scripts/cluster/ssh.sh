#!/usr/bin/env bash
# LAPTOP: reach the DeltaAI login node over the shared master connection (docs/CLUSTER.md).
#   scripts/cluster/ssh.sh                          interactive shell
#   scripts/cluster/ssh.sh scripts/cluster/X.sh     run a repo script there (bash -l, from stdin)
#   scripts/cluster/ssh.sh -O exit                  close the master connection
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=laptop.sh
source "$here/laptop.sh"
if [[ $# -eq 0 ]]; then
  exec ssh "${DTAI_SSH_OPTS[@]}" "$DTAI_HOST"
elif [[ $1 == -O ]]; then
  exec ssh "${DTAI_SSH_OPTS[@]}" -O "${2:-exit}" "$DTAI_HOST"
else
  exec ssh "${DTAI_SSH_OPTS[@]}" "$DTAI_HOST" 'bash -l -s' <"$1"
fi
