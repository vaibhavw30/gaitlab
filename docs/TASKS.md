# Tasks

Work top to bottom. A task is done when its listed tests pass and any listed result file exists. "Scaffolded" means the module and tests already exist in the initial commit and pass on synthetic data; review them, then build on them.

Legend: `[x]` done · `[~]` scaffolded, needs review against real data · `[ ]` to do

---

## Setup (first session with Claude Code)

The scaffold was tested with pytest on synthetic data (71 tests passing, Python 3.13, CPU). It has **not** been through `uv` or Ruff yet.

- [ ] `git init && git add -A && git commit -m "scaffold"` (before any tool rewrites files)
- [ ] `uv python install 3.11 && uv sync --extra video --extra ml --extra viz --extra dev && uv lock`
- [ ] `uv run ruff check --fix . && uv run ruff format .`, then `uv run pytest -m "not gpu and not data"`; commit
- [ ] `uv run gaitlab --help` and `uv run gaitlab synth --n-sessions 6` work
- [ ] Create the GitHub repo, push, confirm CI is green (user task: repo name, visibility, description)

## M0 — Stand-in data (before your own recordings)

Goal: the whole pipeline runs on other people's running and on synthetic data, so the code is debugged before your labels exist.

- [~] **Core types and config** — `types.py`, `config.py`, `configs/*.yaml`. Tests: `test_config.py`
- [~] **Canonical skeleton + backend maps** — `pose/skeleton.py`. Tests: `test_skeleton.py`
- [~] **Synthetic runner with ground truth** — `data/synthetic.py`. Tests: `test_synthetic.py`
- [~] **Signal cleaning** — `signal/clean.py`. Tests: `test_clean.py`
- [~] **Events (IC/TO)** — `gait/events.py`. Tests: `test_events.py` (exact IC on clean synthetic, ±2 frames on noisy)
- [~] **Per-stride features** — `gait/features.py`. Tests: `test_features.py` (vs synthetic truth)
- [~] **Windows** — `gait/windows.py`. Tests: `test_windows.py`
- [~] **Metrics** — `eval/metrics.py`. Tests: `test_metrics.py`
- [~] **Results provenance + README table + claim check** — `report/results.py`, `report/readme_table.py`, `scripts/check_readme_numbers.py`. Tests: `test_results.py`
- [~] **Prompt, SFT builder, verifier, rules, splits** — `llm/prompt.py`, `llm/sft_data.py`, `llm/verifier.py`, `models/rules.py`, `data/splits.py`. Tests: `test_prompt.py`, `test_verifier.py`, `test_splits.py`
- [ ] **Video decode** — `io/video.py` (PyAV, PTS timestamps, fps check). Test: `test_video.py` with a tiny generated clip (PyAV can encode one in-test)
- [ ] **MediaPipe backend** — `pose/mediapipe_backend.py` (Tasks API `PoseLandmarker`, VIDEO running mode, heavy model). Smoke test on a Mixkit clip (`@pytest.mark.data`)
- [ ] **RTMPose backend** — `pose/rtmpose_backend.py` via rtmlib `Body` (Halpe-26 / body7 model). Same smoke test
- [ ] **Overlay** — `viz/overlay.py`: skeleton, contact markers, knee/shin angle text → mp4 and GIF
- [ ] **CLI wiring** — `extract`, `process`, `overlay`, `synth` in `cli.py`
- [ ] **Synthetic dry run** — `gaitlab synth --n-sessions 6` → strides, windows, rule labels (`synthetic=True`) → `baselines` → `eval` writes to `results/synthetic/`
- [ ] **Mocap projection** — `data/mocap.py` for Fukuchi 2017 (smallest; CC-BY) then Running Injury Clinic. → `results/validation/mocap_features.json` (V3)
- [ ] **BioCV loader + event validation** — `data/biocv.py`; `gaitlab validate biocv` → `results/validation/biocv_events.json` (V1), `biocv_angles.json` (V2). **Confirm the license with the user first.**
- [ ] **Read Stenum et al.'s event code; compare with ours on BioCV**, and record the differences in an ADR if we change the method
- [ ] **Pick the primary pose backend** from V1/V2 → ADR-0005 decision filled in
- [ ] Email the Ole Miss dataset author (user task)

## M1 — Capture and signal (own data)

- [ ] Record 3 sessions per `docs/DATA.md §2`; fill `data/sessions.csv` (user task)
- [ ] `extract` + `process` on own sessions with both backends
- [ ] Standing-clip calibration for foot-strike angle (`features.foot_strike_offset_from_standing`)
- [ ] Cadence vs hand counts → `results/validation/own_cadence.json` (V4)
- [ ] First real overlay GIF in README

## M2 — Features and labels

- [ ] Remaining sessions recorded (6–8 total) (user task)
- [ ] Direction checks → `results/validation/directions.json` (V5); stability ICC → `stability.json` (V6)
- [ ] Backend agreement → `results/validation/backend_agreement.json` (V7)
- [ ] **Calibrate `configs/features.yaml` thresholds** from natural-condition windows + Running Injury Clinic percentiles → ADR-0009
- [ ] Labeling app `label/app.py` (normal + blind modes; append-only CSV with lock)
- [ ] Label ~200 windows (user task); freeze `data/splits.json`
- [ ] Baselines `models/baselines.py` (LR, LightGBM, LOSO CV)

## M3 — Grader

- [ ] `llm/grade.py`: chat-template rendering, single-token assertion for " 1"…" 5", logit scoring, greedy tags + cue
- [ ] `llm/train_lora.py` (TRL SFTTrainer, assistant-only loss) for `gemma-3-1b-it` and `gemma-3-270m-it` (`@pytest.mark.gpu` smoke test with 4 steps)
- [ ] Zero-shot 1B predictions
- [ ] Blind re-grade of 40 windows (user task) → `results/grader/intra_rater.json` (G2)
- [ ] `eval/run_eval.py` → G1, G3–G6 result files
- [ ] `viz/plots.py`: ladder chart, knee-angle-by-condition, confusion matrix, reliability diagram

## M4 — Speed and polish

- [ ] `bench/latency.py` → `results/bench/stages.json` (B1)
- [ ] `bench/trt.py` (Torch-TensorRT scoring wrapper, parity gate) → `results/bench/trt.json` (B2)
- [ ] Nsight Systems trace screenshot in `results/bench/`
- [ ] `app/coach.py` Gradio demo
- [ ] `gaitlab report` fills README results; `check_readme_numbers.py` clean
- [ ] Repo description and topics set on GitHub (user task)

## M5 — Stretch

- [ ] RTMPose TensorRT FP16 engine + bench
- [ ] Second validation set (Guo et al. 2026 or AthletePose3D), license confirmed first
- [ ] MCP server: `analyze_session`, `compare_sessions`, `get_norms`
- [ ] Population percentiles (`gait/norms.py`) in the app
