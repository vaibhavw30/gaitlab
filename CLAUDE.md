# CLAUDE.md — GaitLab

GaitLab grades treadmill running form from a side-view phone video. The pipeline is pose keypoints → deterministic per-stride biomechanics → a LoRA-tuned small LLM (Gemma 3 1B / 270M) that outputs a 1–5 grade (read from logits), issue tags, and one cue. A rule-based verifier checks the cue. TensorRT accelerates the GPU hot spots.

Read before non-trivial work: `docs/TECHNICAL_PLAN.md` (contracts and algorithms), `docs/TASKS.md` (what's next), `docs/EVALUATION.md` (how numbers are made). Product context: `docs/PRD.md`. Decisions: `docs/adr/`.

## Commands

```bash
uv sync --extra video --extra ml --extra viz --extra dev        # CPU dev setup
uv sync --all-extras                                             # full GPU box (see README for torch index)
uv run pytest -m "not gpu and not data"                          # what CI runs
uv run pytest tests/test_events.py -q                            # one module
uv run ruff check . && uv run ruff format .
uv run python scripts/check_readme_numbers.py                    # no stray result numbers in README
uv run gaitlab --help                                            # CLI (src/gaitlab/cli.py)
uv run gaitlab synth --n-sessions 6                              # synthetic data for dry runs
```

## Layout

- `src/gaitlab/` — `signal/`, `gait/`, `eval/metrics.py`, `llm/prompt.py` and `llm/verifier.py` are **pure** (arrays in, arrays/dataclasses out, no I/O). File I/O lives in `io/`, `data/`, `report/`, `cli.py`.
- `configs/*.yaml` — every run is driven by one config; its hash goes into results.
- `data/` — git-ignored. Never commit anything from it. Layout in `docs/DATA.md`.
- `results/` — committed JSON written **only** by `report/results.py`. `results/synthetic/` is never reported.
- `tests/` — mirrors `src/`. The fixture is `gaitlab.data.synthetic` (a runner with known ground truth).

## Hard rules (these override everything else)

1. **No invented numbers.** Never write a metric, speedup, count or accuracy into README, docs, docstrings, commit messages or the app unless it was read from a `results/*.json` file with `synthetic: false` and `git_dirty: false`. If a number doesn't exist yet, write `TBD (results/<area>/<name>.json)`.
2. **Results only via `report.results.write_result()`.** Don't hand-edit result JSON. Don't hard-code results in `report/readme_table.py`.
3. **Synthetic stays synthetic.** Anything produced from `data/synthetic.py`, `gaitlab synth`, or rule-generated labels sets `synthetic=True`.
4. **Split by session.** Never random-split windows. `data/splits.json` is frozen once written. Don't tune anything on `test`.
5. **Thresholds are not guessed.** `configs/features.yaml` thresholds stay `null` until calibrated per `TECHNICAL_PLAN §5.5`, and the calibration gets an ADR. Rules and the verifier must raise `ThresholdsNotCalibrated` on real data while they're null.
6. **Public data:** don't download, vendor or commit datasets on the user's behalf without them confirming the license. Don't copy code from GPL projects (e.g. OpenGait) into this repo.
7. **Claims about tools:** the README lists only libraries actually imported by code that runs.
8. **Stay in this repository.** Only read and write files inside this repository. Never create, move or edit files in other directories (e.g. ~/vip-tcn-training). If something seems to belong elsewhere, stop and ask.

## Conventions

- Python 3.11, full type hints, NumPy-style docstrings with units (px, s, deg, L = leg lengths).
- Coordinates: stored y-down pixels; convert once with `gait.geometry.to_y_up`. Angles in degrees. `d` = facing direction (+1 right, −1 left).
- Canonical skeleton: `pose.skeleton.CANONICAL` (13 joints). Index by name via `J["l_heel"]`, never by magic number.
- Prefer small pure functions plus a thin orchestrator. Return dataclasses from `gaitlab.types`, not dicts.
- Tests: write or extend the test first, then implement. Mark GPU tests `@pytest.mark.gpu`, tests needing downloaded data `@pytest.mark.data`.
- New config fields: add to the pydantic model in `config.py` **and** the YAML default **and** TECHNICAL_PLAN.
- When code and `docs/TECHNICAL_PLAN.md` disagree, update one of them in the same change.

## Workflow

1. Pick the next unchecked item in `docs/TASKS.md` (or run `/milestone M<n>`).
2. Read the relevant TECHNICAL_PLAN section. Write or extend tests. Implement.
3. Run `uv run pytest -m "not gpu and not data"` and `uv run ruff check .`. Both must be clean.
4. Tick the box in `docs/TASKS.md`. If a decision was made, add an ADR (`/adr <title>`).
5. Never mark a task done with failing tests or unwritten result files.

## Gotchas

- MediaPipe `foot_index` is the toe. Halpe-26 heel is index 24/25, big toe 20/21. COCO-17 has no feet: foot features become NaN, and that's expected.
- Far-side limbs in a side view are unreliable. Features are computed on the **near side** (higher visibility).
- Phone video can have variable frame rate. Use container PTS (`t` array), not `frame / fps`, for timing.
- Gemma grade tokens: assert `" 1"…" 5"` are single tokens for the loaded tokenizer before trusting logit grading. Gemma 3 splits them (`" "` + `"1"`), so `llm/grade.py` scores the bare digits after `"grade: "` (ADR-0013).
- torch and LightGBM segfault when loaded into one process on macOS (separate OpenMP runtimes). Keep `baselines` and `train`/`predict` in separate processes; tests chaining them use subprocesses.
- Model weights live in `.hf_cache/` inside the repo (rule 8); `llm.runtime` passes that `cache_dir`. Don't download into `~/.cache/huggingface`.
- Torch-TensorRT: compile only the fixed-shape scoring wrapper (ADR-0004), not `generate()`.
