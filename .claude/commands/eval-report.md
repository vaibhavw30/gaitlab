---
description: Run evaluation and regenerate the README results table
---

1. Confirm `data/splits.json` exists and hasn't changed since the last results. Compare its hash with the `data_manifest_hash` in the most recent `results/grader/*.json`. If it changed, stop and tell me.
2. Run `uv run gaitlab eval` and `uv run gaitlab report`.
3. Show the diff of `README.md` between the RESULTS markers and list every result file touched, with its `synthetic` and `git_dirty` flags.
4. If any result has `git_dirty: true`, tell me to commit first and rerun. Don't commit the README change yourself.
5. Run `/check-claims`.
