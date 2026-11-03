---
description: Audit every number in README, docs and app strings against results/*.json
---

Audit claims in this repo.

1. Run `uv run python scripts/check_readme_numbers.py` and report its output.
2. Search `README.md`, `docs/`, `src/gaitlab/app/` and `src/gaitlab/viz/` for result-like numbers: percentages, "x faster", kappa, QWK, MAE, RMSE, F1, fps achieved, ms latency, counts of clips, strides or labels.
3. For each one, find the `results/**/*.json` file and key it came from. Check `synthetic: false` and `git_dirty: false`.
4. Output a table: location · claim · source file/key · status (OK / NO SOURCE / SYNTHETIC / DIRTY / STALE, where stale means the value differs from the JSON).
5. Don't edit anything. Propose fixes, replacing unsupported numbers with `TBD (results/<area>/<name>.json)`.

Design parameters (20-stride windows, 120 fps capture, LoRA rank 16, model names) are not claims; skip them.
