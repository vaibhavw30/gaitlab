#!/usr/bin/env bash
# LAPTOP: send code, configs, .git and data/synthetic to ~/gaitlab on DeltaAI (docs/CLUSTER.md step 1).
#   scripts/cluster/rsync-up.sh --dry-run   list what would be sent and check it; sends nothing
#   scripts/cluster/rsync-up.sh             the same check, then the transfer
# Never sent: .venv, .hf_cache, runs, results, models, logs, .env, tool caches, and everything
# under data/ except data/synthetic. The check below refuses to transfer if any of those appear.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=laptop.sh
source "$here/laptop.sh"
cd "$(git -C "$here" rev-parse --show-toplevel)"

dry=0
case "${1:-}" in
  --dry-run) dry=1 ;;
  "") ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
dest="${GAITLAB_RSYNC_DEST:-$DTAI_HOST:gaitlab/}" # override only in tests (a local directory)

FILTERS=(
  --include=/data/ --include=/data/synthetic/ --include=/data/synthetic/** --exclude=/data/*
  --exclude=/.venv/ --exclude=/.hf_cache/ --exclude=/runs/ --exclude=/results/
  --exclude=/models/ --exclude=/logs/
  --exclude=.env --exclude=.env.* --exclude=__pycache__/ --exclude=.pytest_cache/
  --exclude=.ruff_cache/ --exclude=.mypy_cache/ --exclude=.DS_Store
)

listing="$(rsync -a --dry-run -v "${FILTERS[@]}" -e "$DTAI_RSYNC_SSH" ./ "$dest")"
bad="$(printf '%s\n' "$listing" | awk '
  /^data\// && $0 != "data/" && $0 !~ /^data\/synthetic(\/|$)/ { print; next }
  /^(\.venv|\.hf_cache|runs|results|models|logs)(\/|$)/ { print; next }
  /(^|\/)\.env(\.|$)/ { print }')"
if [[ -n $bad ]]; then
  printf 'REFUSING: these paths must not leave the laptop:\n%s\n' "$bad" >&2
  exit 1
fi

if ((dry)); then
  printf '%s\n' "$listing"
  echo "CHECK OK: nothing from data/ except data/synthetic, no caches, runs, results or .env."
  echo "Dry run only: nothing sent."
  exit 0
fi
echo "CHECK OK; sending."
rsync -av "${FILTERS[@]}" -e "$DTAI_RSYNC_SSH" ./ "$dest"
