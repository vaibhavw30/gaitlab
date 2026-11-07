#!/usr/bin/env bash
# LAPTOP: bring back one run's small artifacts and the rf_* job logs (docs/CLUSTER.md step 7).
#   scripts/cluster/rsync-down.sh <run_id> [--with-results]
# Pulls runs/<run_id>/{adapter/, trainer_state.json, run.json, config.json, SYNTHETIC} and
# logs/rf_*.out. --with-results also pulls results/synthetic/**/*.json. Never model weights, caches
# or checkpoints: anything not listed is excluded.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=laptop.sh
source "$here/laptop.sh"

run="${1:-}"
if [[ ! $run =~ ^(1b|270m)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$ ]]; then
  echo "usage: $0 <run_id like 1b-20261104T120000Z-0123abcd> [--with-results]" >&2
  exit 2
fi
with_results=0
case "${2:-}" in
  --with-results) with_results=1 ;;
  "") ;;
  *) echo "unknown option ${2}" >&2; exit 2 ;;
esac
src="${GAITLAB_RSYNC_SRC:-$DTAI_HOST:gaitlab}"                        # override only in tests
dst="${GAITLAB_LOCAL_ROOT:-$(git -C "$here" rev-parse --show-toplevel)}" # override only in tests
mkdir -p "$dst/runs/$run" "$dst/logs"

rsync -av -e "$DTAI_RSYNC_SSH" \
  --include=/adapter/ --include=/adapter/* \
  --include=/trainer_state.json --include=/run.json --include=/config.json --include=/SYNTHETIC \
  --exclude=* \
  "$src/runs/$run/" "$dst/runs/$run/"
rsync -av -e "$DTAI_RSYNC_SSH" --include=rf_*.out --exclude=* "$src/logs/" "$dst/logs/"
if ((with_results)); then
  mkdir -p "$dst/results/synthetic"
  rsync -av -e "$DTAI_RSYNC_SSH" --include=*/ --include=*.json --exclude=* \
    "$src/results/synthetic/" "$dst/results/synthetic/"
fi
du -sh "$dst/runs/$run"
