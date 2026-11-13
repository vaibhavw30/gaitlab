# GaitLab — Technical Implementation Plan

Companion to [PRD.md](PRD.md). This document is the contract Claude Code implements against: module boundaries, data schemas, algorithms, and acceptance criteria per milestone. When code and this document disagree, fix one of them in the same PR.

---

## 1. Architecture

```
                 ┌──────────────── runtime pipeline ────────────────┐
 video.mp4 ──► io.video ──► pose.{mediapipe,rtmpose} ──► signal.clean ──► gait.events ──► gait.features ──► gait.windows
   (PyAV)       frames+ts      KeypointSeq (canonical)    gated+filtered     IC / TO          per-stride        20-stride summaries
                                                                                                                    │
                                                         llm.grade (Gemma 3 + LoRA, logits) ◄── llm.prompt ◄────────┤
                                                              │ grade, P(grade), tags, cue                           │
                                                         llm.verifier ──► app.coach (Gradio) / viz.overlay           │
                 └───────────────────────────────────────────────────────────────────────────────────────────────────┘

                 ┌──────────────── training & evaluation ───────────┐
 labels.csv (label.app) ──► data.splits (by session) ──► models.{rules,baselines} ─┐
                                    └──────────────► llm.sft_data ──► llm.train_lora ─┤
                                                                                       ▼
                                                    eval.run_eval ──► report.results ──► results/*.json ──► report.readme_table
                                                    bench.{latency,trt} ───────────────┘
```

Design rules:
- **Pure core.** `signal/`, `gait/`, `eval/metrics.py`, `llm/prompt.py`, `llm/verifier.py` are pure functions over NumPy arrays and dataclasses. No file I/O and no global state, so they're unit-testable on synthetic data.
- **I/O at the edges.** `io/`, `data/`, `report/`, `cli.py` read and write files.
- **One canonical skeleton.** Each pose backend maps to `pose.skeleton.CANONICAL`; nothing downstream knows which backend produced the data.
- **Numbers flow one way.** Metrics → `report.results.write_result()` → `results/*.json` → README table. Nothing else writes a number anyone reads.

## 2. Repository layout

```
gaitlab/
├── CLAUDE.md                      # instructions for Claude Code
├── README.md                      # public face; results table is generated
├── pyproject.toml                 # uv-managed; extras per stage
├── configs/                       # every run = one YAML, hashed into results
│   ├── default.yaml               # pipeline params
│   ├── features.yaml              # feature flag ranges + tag vocabulary (verifier, rules)
│   ├── train_lora.yaml
│   └── bench.yaml
├── data/                          # git-ignored; layout in DATA.md
├── results/                       # committed JSON results (+ results/synthetic/, not reported)
├── docs/                          # PRD, this plan, DATA, EVALUATION, TASKS, ADRs
├── scripts/                       # model download, dataset fetch helpers
├── src/gaitlab/
│   ├── cli.py                     # Typer entry point: `gaitlab <cmd>`
│   ├── config.py                  # pydantic config models + hashing
│   ├── types.py                   # shared dataclasses (KeypointSeq, Stride, Window …)
│   ├── io/video.py                # PyAV decode, metadata, fps checks
│   ├── pose/{base,skeleton,mediapipe_backend,rtmpose_backend}.py
│   ├── signal/clean.py            # gating, interpolation, Butterworth
│   ├── gait/{geometry,events,features,windows,norms}.py
│   ├── data/{sessions,splits,synthetic,biocv,mocap}.py
│   ├── label/app.py               # Gradio labeling app
│   ├── models/{rules,baselines}.py
│   ├── llm/{prompt,sft_data,train_lora,grade,verifier}.py
│   ├── eval/{metrics,run_eval}.py
│   ├── bench/{latency,trt}.py
│   ├── viz/{overlay,plots}.py
│   ├── app/coach.py               # Gradio coach app (demo)
│   └── report/{results,readme_table}.py
└── tests/                         # CPU tests; GPU tests marked @pytest.mark.gpu
```

## 3. Environment

- Linux or WSL2 Ubuntu, NVIDIA driver + CUDA matching the installed PyTorch wheel. TensorRT-LLM dropped Windows support at v0.18 and Torch-TensorRT on Windows is Dynamo-only, so don't target native Windows.
- **GPU host requirement (from `uv.lock`):** the locked Linux torch 2.14.1 wheels (x86_64 and aarch64) are CUDA 13.0 builds: they depend on `cuda-toolkit` 13.0, `nvidia-cudnn-cu13` and `nvidia-nccl-cu13`. TensorRT is `tensorrt-cu13` 11.1, with torch-tensorrt 2.14.0. So the NVIDIA driver must support CUDA 13 (`nvidia-smi` shows "CUDA Version: 13.x"). On aarch64 the TensorRT wheels (`tensorrt-cu13-bindings`, `tensorrt-cu13-libs`) are `manylinux_2_35` only, so they need glibc 2.35 or newer; the `llm` and `dev` extras need at most `manylinux_2_28`. A host that can't meet these runs the GPU stages in an NGC PyTorch container instead ([CLUSTER.md](CLUSTER.md) "Fallback"). DeltaAI is checked by the `rf_gate` job; confirmed versions: TBD (docs/CLUSTER.md "Confirmed on DeltaAI").
- Python 3.11 via `uv`. Base dependencies are NumPy, SciPy, pandas, pyarrow, PyYAML, pydantic, Typer and Rich. Extras: `video`, `mediapipe`, `rtm`, `app`, `ml`, `llm`, `trt`, `viz`, `dev`. CI installs base + `video` + `ml` + `viz` + `dev` only.
- Pin versions with `uv lock` after the first working install. Record versions in every result file (§9).
- Models: MediaPipe `pose_landmarker_heavy.task` (scripts/download_models.sh); `google/gemma-3-1b-it` and `google/gemma-3-270m-it` from Hugging Face (accept the Gemma license first).

## 4. Data contracts

All schemas also live as dataclasses or pydantic models in `types.py` and `data/sessions.py`. Field names below are canonical.

### 4.1 Canonical skeleton (`pose/skeleton.py`)

`nose, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle, l_heel, r_heel, l_toe, r_toe` (13 joints, index order fixed).
- MediaPipe 33 → canonical: heel = 29/30, toe = `foot_index` 31/32.
- Halpe-26 (RTMPose body7) → canonical: heel = 24/25, toe = big toe 20/21.
- COCO-17 sources (e.g. Guo et al. YOLOv8 CSVs) have no heel or toe. Load them with heel and toe = NaN; foot-dependent features become NaN, which is tested.

**Backends** (`pose/base.py` `get_backend`; both behind `PoseBackend.run(frames, fps, size)`):
- `mediapipe`: Tasks API `PoseLandmarker`, VIDEO mode, one pose, `pose.mediapipe_model` (heavy). Timestamps are integer ms from PTS, bumped to stay strictly increasing; `KeypointSeq.t` keeps the true PTS. `vis` = landmark visibility.
- `rtmpose`: rtmlib `BodyWithFeet` model zoo, i.e. a YOLOX person detector plus an RTMPose **Halpe-26** model; `pose.rtmpose_mode` picks `performance` | `balanced` | `lightweight`. Models are downloaded by rtmlib's `download_checkpoint` into `models/rtmlib/` (inside the repo, git-ignored), not `~/.cache`. Detector and pose model are called separately: rtmlib's pose model falls back to the whole image when it gets no boxes, so a frame where the detector finds nobody is NaN with vis 0, not a full-frame skeleton. Several people → the one with the highest mean keypoint score. Frames go in as BGR. `vis` = SimCC score clipped to [0, 1]; raw p05/p50/p95 in `meta["score_quantiles"]` for tuning `clean.visibility_min` per backend. onnxruntime runs on `cuda` when its CUDA provider exists, else `cpu`. A model without 26 keypoints is refused.
- Both: no detection → NaN xy, vis 0; frame size must match the clip; meta records the library version and model file hashes.

### 4.2 `KeypointSeq` (`data/interim/<session>/<clip>.<backend>.npz`)

| key | shape | dtype | meaning |
|---|---|---|---|
| `xy` | (T, 13, 2) | float32 | pixels, origin top-left, **y down** as stored |
| `vis` | (T, 13) | float32 | visibility or confidence in [0, 1] |
| `t` | (T,) | float64 | seconds from the first presented frame, from container PTS × time_base ([ADR-0010](adr/0010-video-decode-timestamps-rotation-vfr.md)) |
| `fps` | () | float64 | nominal fps |
| `size` | (2,) | int32 | (width, height) |
| `backend` | () | str | `mediapipe` / `rtmpose` / `mocap_proj` / `synthetic` |
| `meta` | () | str (JSON) | backend version, model file hash; from `extract` also `session_id`, `clip_id`, `video`, `video_sha256`, `vfr`, `codec` |

Downstream code converts to **y-up** once via `gait.geometry.to_y_up()`.

### 4.3 Strides (`data/processed/<session>/strides.parquet`)

One row per near-side stride: `session_id, clip_id, condition, stride_idx, side, ic_frame, to_frame, next_ic_frame, t_ic, stride_time_s, contact_time_s, duty_factor, cadence_spm, overstride_L, shin_angle_deg, knee_flex_ic_deg, knee_flex_peak_deg, foot_strike_deg, foot_strike_calibrated (bool), vert_osc_L, trunk_lean_deg, quality, knee_flex_cycle_deg (101 floats: knee flexion from IC to next IC, resampled by time; `features.knee_cycle_curve`), synthetic (bool)`. `synthetic` is True when the KeypointSeq was generated (`KeypointSeq.synthetic`); such outputs also get a `SYNTHETIC` marker file in their folder ([ADR-0011](adr/0011-no-video-path-synthetic-keypoints-single-clip-overlay.md)).

### 4.4 Windows (`data/processed/<session>/windows.parquet`)

`window_id = f"{session_id}:{clip_id}:{start_stride}"`, `session_id, clip_id, condition, start_stride, n_strides, t_start, t_end, speed_kmh, quality`, then `<feature>_med` and `<feature>_iqr` for each feature in §5.3, then `synthetic` (bool, as in §4.3).

### 4.5 Labels (`data/labels/labels.csv`)

`window_id, grade (1–5), tags (semicolon-separated from features.yaml vocabulary), cue (free text), rater, mode (normal|blind_regrade), labeled_at (ISO 8601)`.

### 4.6 Sessions (`data/sessions.csv`)

`session_id, date, clip_id, path, condition (natural|overstride|metronome|lean|fatigue|standing), speed_kmh, incline_pct, shoes, fps, camera_height_m, camera_distance_m, notes`.

### 4.7 SFT JSONL (`data/sft/{train,dev,test}.jsonl`)

`{"window_id", "messages": [{"role":"user","content": <prompt>}, {"role":"assistant","content": <target>}], "synthetic": bool}`

### 4.8 Results (`results/<area>/<name>.json`) — see §9.

## 5. Algorithms

All angles in degrees. `d ∈ {+1, −1}` is the direction the runner faces in the image: `d = sign(median(toe_x − heel_x))` on the near side, with y-up coordinates.

### 5.1 Cleaning (`signal/clean.py`)

1. **Near side:** the side with higher mean visibility over {hip, knee, ankle, heel, toe}.
2. **Gating:** set a joint to NaN where `vis < cfg.visibility_min` (default 0.5).
3. **Interpolation:** linear over NaN runs no longer than `cfg.max_gap_s` (default 0.025 s, about 3 frames at 120 fps). Longer gaps stay NaN.
4. **Filter:** zero-phase 4th-order Butterworth (`scipy.signal.butter` + `filtfilt`), applied per contiguous non-NaN segment; segments too short for `filtfilt` padding stay unfiltered. Two cutoffs ([ADR-0008](adr/0008-filter-cutoffs-events-vs-features.md)): `cfg.lowpass_hz` (default 12 Hz) for the signal used to detect events, and `cfg.feature_lowpass_hz` (default 20 Hz) for the signal features are measured on. Heavier filtering delays detected contact and biases contact-instant features.

### 5.2 Events (`gait/events.py`)

Per foot, on y-up data:
1. `foot_low(t) = min(heel_y, toe_y)`; `ground = percentile(foot_low, 5)`.
2. `L` = leg length = median over frames of `|hip−knee| + |knee−ankle|` on the near side.
3. Candidate contact where `foot_low < ground + cfg.contact_height_frac × L` (default 0.03) **and** the foot moves backward with the belt: `d × vx_heel < 0` (vx from a central difference on the filtered signal).
4. Contact runs shorter than `cfg.min_stance_s` (default 0.08 s) are discarded. **Initial contact (IC)** = first frame of a run; **toe-off (TO)** = last frame.
5. Strides = consecutive ICs on the same foot. Discard strides with `stride_time ∉ [cfg.stride_time_min_s, cfg.stride_time_max_s]` (defaults 0.45 s and 1.0 s).

Return `Events(ic: np.ndarray[int], to: np.ndarray[int], side: str)`.
Validation (M0): IC error vs BioCV force-plate contacts, reported in frames and ms. Before tuning, read Stenum et al.'s event code ([link](https://github.com/janstenum/GaitAnalysis-PoseEstimation)).

### 5.3 Features (`gait/features.py`)

For each near-side stride [IC_k, IC_{k+1}):

| Feature | Formula |
|---|---|
| `stride_time_s` | `(IC_{k+1} − IC_k) / fps` (use `t` arrays when fps is variable) |
| `cadence_spm` | `120 / stride_time_s` |
| `contact_time_s` | `(TO_k − IC_k + 1) / fps` |
| `duty_factor` | `contact_time_s / stride_time_s` |
| `overstride_L` | `d × (heel_x − hip_x)[IC] / L` (positive = foot lands ahead of hip) |
| `shin_angle_deg` | `atan2(d × (ankle_x − knee_x), knee_y − ankle_y)` at IC (positive = shin reaching forward) |
| `knee_flex_ic_deg` | `180 − ∠(hip, knee, ankle)` at IC |
| `knee_flex_peak_deg` | max knee flexion over [IC_k, TO_k] |
| `foot_strike_deg` | `atan2(toe_y − heel_y, d × (toe_x − heel_x))` at IC **minus** the standing angle from the session's `standing` clip. `foot_strike_calibrated = False` if there is no standing clip |
| `vert_osc_L` | `(max − min)(hip_y)` over the stride `/ L`; also `vert_osc_cm = vert_osc_L × leg_length_cm` if the runner's leg length is configured |
| `trunk_lean_deg` | `atan2(d × (shoulder_x − hip_x), shoulder_y − hip_y)` averaged over stance (positive = forward) |
| `quality` | fraction of stride frames where all needed joints are non-NaN |

Strike type (rules and display only): rearfoot > 8°, midfoot −1.6° to 8°, forefoot < −1.6° (Altman & Davis 2012; calibrated angle only).

### 5.4 Windows (`gait/windows.py`)

Non-overlapping runs of `cfg.window_strides` (default **20**, per Riazati et al. 2019) consecutive valid strides inside one clip. Drop a window if `mean(quality) < cfg.window_quality_min` (default 0.8). Summaries: median and IQR per feature.

### 5.5 Rules baseline and verifier ranges (`configs/features.yaml`)

Each tag maps to one feature condition, e.g. `overstride: overstride_L_med > θ`. The **θ values ship as `null` placeholders.** They are set in M2 from the Running Injury Clinic population percentiles and your own natural-condition windows, and recorded in an ADR. Until then, rules and the verifier refuse to run on real data (they raise `ThresholdsNotCalibrated`). Synthetic tests use test-local thresholds.

Tag vocabulary (v1), defined in `configs/features.yaml` with each tag's feature, operator and cue keywords: `overstride, low_cadence, high_bounce, stiff_knee_landing, steep_heel_strike, forward_lean, backward_lean`.

## 6. Labeling (`label/app.py`)

Gradio app, served on 127.0.0.1 only. Left: the window's clip segment `[t_start, t_end]` with the skeleton overlay (`viz.overlay.render_overlay(t_range=…)`; events and angles come from the whole clip), on the source video when it's found (`sessions.csv` path, else `raw/<session>/<clip>.mp4`), else on a plain canvas. Clips are rendered on first view into `<root>/labels/overlays/` and reused. Right: feature medians and IQRs, a 1–5 radio, tag checkboxes (`features.yaml` vocabulary), a cue textbox, Back / Skip / Save. Keys 1–5 set the grade and Enter saves (not while typing in a text box). Logic lives in `label/queue.py` (pure); the writer is `data.sessions.append_label`.
- **Normal mode** shows windows without a normal-mode label, ordered by session, then clip (windows-table order), then `t_start`.
- **Blind re-grade mode** (`--blind N`, default 40 in EVALUATION G2) draws `N` normal-labeled windows with `--seed` (from the sorted ids, so the draw depends only on which windows are labeled) and skips those already re-graded, so a restart resumes the same sample. It hides previous labels, the feature table, the window id (a clip id names the condition) and the overlay's angle/cadence text. Rows are written with `mode=blind_regrade`.
- Each Save validates the row (`LabelRow`: grade 1–5, tags from the vocabulary in vocabulary order, cue collapsed to one line, rater, UTC `labeled_at`) and appends it under an exclusive `fcntl.flock`; the header is written once; a file with a different header is refused. Earlier rows are never rewritten: a correction is a new row and the later one wins (`latest_normal_labels`).
- Synthetic windows can only be labeled with `--synthetic`, into `data/synthetic/labels/hand_labels.csv` (never synth's rule-label `labels.csv`); real and synthetic windows are never mixed (CLAUDE.md rule 3).

## 7. Models

### 7.1 Baselines (`models/`)

- `rules.py`: grade = 5 − (number of flagged tags), clipped to [1, 5]. Tags = flagged tags.
- `baselines.py`: logistic regression (multinomial, standardized features) and LightGBM (multiclass, small fixed grid). Leave-one-session-out CV on train + dev; final fit on train + dev, scored on test. `predict_proba` always returns 5 columns (grades 1–5; unseen grades get 0). Tags: one-vs-rest LR per tag, shared by both. Predictions are written as `<model>.csv` with `window_id, pred_grade, p1..p5, tags, cue, synthetic` ([ADR-0012](adr/0012-synthetic-grader-dry-run.md)).

### 7.2 LLM grader (`llm/`)

**Prompt** (`prompt.py`, deterministic, unit-tested):
```
Grade this treadmill running window from 1 (poor) to 5 (excellent).
speed_kmh: 12.0
cadence_spm: 164.0 (iqr 2.1)
overstride_L: 0.28 (iqr 0.03)
...one line per feature, fixed order, fixed precision...
```
**Target:**
```
grade: 3
tags: overstride; low_cadence
cue: <one sentence>
```
Messages go through the model's chat template (user = prompt, assistant = target).

**Grade by logits** ([ADR-0002](adr/0002-grade-from-logits.md), [ADR-0013](adr/0013-llm-grader-on-cpu-mps-and-gemma3-grade-tokens.md)): render the chat with the assistant turn prefilled with `"grade:"`. Run one forward pass and take next-token logits at the last position, restricted to the token ids of `" 1"…" 5"`. Assert each is a single token for the tokenizer (unit test); if not, score the bare digits `"1"…"5"` after `"grade: "` instead (Gemma 3 does this, since it splits `" 3"` into `" "` + `"3"`). Either way, assert that the digit follows the prefix as one token inside the real chat template. Both checks run before the weights load. Softmax → `P(grade)`. Report argmax and expected grade. Tags and cue come from a separate greedy `generate` continuing after the grade.

**Device** (`llm/runtime.py`): `cuda` > `mps` > `cpu` (`--device` overrides). CUDA loads bf16; MPS and CPU load float32 with eager attention, and training always uses eager attention. Model files go in the repo-local Hugging Face cache `.hf_cache/` unless `HF_HOME` is set. torch and LightGBM never share a process (they segfault together on macOS; ADR-0013).

**Fixed shape for TensorRT:** prompts are padded (left) to `cfg.max_prompt_tokens` so the scoring pass has a static shape ([ADR-0004](adr/0004-tensorrt-placement.md)).

**LoRA** (`train_lora.py`, `configs/train_lora.yaml`, typed by `config.TrainLoraConfig`): PEFT LoRA r=16, α=32, dropout 0.05, targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`; TRL `SFTTrainer` with `assistant_only_loss` (TRL swaps in its Gemma 3 training template, which marks the assistant turn including `<end_of_turn>`); lr 2e-4 cosine, warmup 5%, 3 epochs, bf16 on CUDA (float32 elsewhere), effective batch 16, seed 0, eval on dev each epoch, best checkpoint by dev loss. `--max-steps` caps steps for smoke runs. Same recipe for `gemma-3-1b-it` and `gemma-3-270m-it`. `runs/<run_id>/` (`<model_key>-<UTC time>-<config hash>`) holds `adapter/`, `config.json`, `run.json` (config and data hashes, device, dtype, versions, `synthetic`), `trainer_state.json` with timing entries removed, and a `SYNTHETIC` marker for dry runs. Synthetic SFT records need `--allow-synthetic`; a mix of synthetic and real records is always refused. W&B logging is optional and CUDA-only.

**Predict** (`gaitlab predict`): for each labeled test window, logit grade + `p1..p5`, then greedy tags and cue, written as `predictions/gemma3_<key>_{lora,zeroshot}.csv` (§7.1 schema) with `meta.json` (`generates_cues`, grade-token form, run id, device). G6 repeats (`consistency.runs` scoring passes; tags and cue sampled at `consistency.temperature`, no top-k/top-p) go to `<model>.consistency.json`.

### 7.3 Cue verifier (`llm/verifier.py`)

Input: window features, generated tags, cue, `features.yaml`. Pass iff (a) every generated tag is in the vocabulary, (b) every generated tag's condition holds for this window, and (c) the cue mentions at least one keyword of a generated tag. With no generated tags, the cue must not mention any tag's keywords (it can't imply an issue the model didn't flag). Returns `VerifierResult(passed, reasons[])`. Failed cues are suppressed in the app and counted in eval. Raises `ThresholdsNotCalibratedError` while a needed threshold is null.

## 8. Evaluation and benchmarks

Metric definitions, splits and protocol live in [EVALUATION.md](EVALUATION.md). Implementation:
- `eval/metrics.py`: exact agreement, within-1, quadratic-weighted kappa, bootstrap CI (resample **sessions**, not windows), macro-F1 for multi-label tags, ECE (10 bins) and Brier on grade probabilities, ICC(2,1), Bland–Altman bias and limits, MAE.
- `eval/run_eval.py`: loads predictions from each model, computes everything, writes `results/grader/*.json`: `ladder`, `tags`, `calibration` for every model; `verifier` (G4: pass rate with session-bootstrap CI and a histogram of failure codes `unknown_tag | unsupported_tag | cue_misses_tags | cue_implies_untagged`) for models whose `meta.json` says `generates_cues`; `consistency` (G6: share of windows with identical grade over the scoring repeats, and identical tags / cue over the sampled repeats) for models with a `<model>.consistency.json`.
- `bench/latency.py`: CUDA-event timing with warmup and iterations from `bench.yaml`; reports p50/p95/mean per stage (decode, pose, clean+events+features, scoring pass, generate) and end-to-end fps on a fixed clip; peak VRAM via `torch.cuda.max_memory_allocated`. `cuda_time(fn, warmup, iters)` is written (events around each call, synchronize before reading, peak memory reset after warmup, raises without CUDA); `bench_pipeline` is not yet.
- `bench/trt.py`: Torch-TensorRT `compile(ir="dynamo", use_explicit_typing=True)` on a wrapper module returning the 5 grade logits for a padded batch, one static-shape module per batch size. Compare against eager at the same precision (`bench.yaml precision`, bf16 by default because Gemma 3's residual stream overflows FP16 and gives non-finite grade logits, [ADR-0015](adr/0015-bench-precision-bf16.md); `fp16` stays selectable) at batch 1 and 16. Prompts are the SFT `test.jsonl` user turns, cut into static batches (the last one padded by repeating its last prompt; padding rows dropped). **Parity gate:** argmax grade identical on 100% of test prompts at every batch size (the worst batch size is reported), max |ΔP| reported, and any non-finite output counts as a disagreement. If parity fails, the speedup fields are omitted and `speedup_reported` is false ([ADR-0014](adr/0014-tensorrt-bench-parity-details.md)). `gaitlab bench --trt [--synthetic]` writes `results/[synthetic/]bench/trt.json`; it refuses a synthetic/real mismatch between `--synthetic`, the test prompts and the adapter run.

## 9. Results and provenance (`report/results.py`)

`write_result(area, name, metrics: dict, *, config, data_manifest, synthetic: bool)` writes:
```json
{
  "schema": 1,
  "area": "grader", "name": "qwk_ladder",
  "synthetic": false,
  "created_at": "...",
  "git_sha": "...", "git_dirty": false,
  "config_hash": "sha256:...", "data_manifest_hash": "sha256:...",
  "hardware": {"gpu": "...", "driver": "...", "cuda": "...", "cpu": "..."},
  "versions": {"python": "...", "torch": "...", "tensorrt": "...", "mediapipe": "...", ...},
  "metrics": { ... }
}
```
- `synthetic=True` routes to `results/synthetic/` and is **never** read by `readme_table.py`.
- `git_dirty=True` results are allowed locally but CI and the README generator reject them.
- `report/readme_table.py` rewrites only the block between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` in README.md.

## 10. CLI (`gaitlab …`)

| Command | Does |
|---|---|
| `extract --session S [--sessions CSV] [--backend {mediapipe,rtmpose}] [--clip C] [--allow-low-fps] [--overwrite]` | video → KeypointSeq npz for every clip of S (standing clips too), written to `<CSV dir>/interim/S/<clip>.<backend>.npz` where `process` reads it. Backend defaults to `pose.backend`. Every clip is probed and fps-checked (`capture.min_fps`) before any model runs; rows that already point at an `.npz` are refused (use `process`); existing npz are skipped unless `--overwrite` |
| `process --session S [--sessions CSV]` | clean → events → strides.parquet → windows.parquet for every clip of S. Reads `<CSV dir>/interim/S/<clip>.<backend>.npz`, writes `<CSV dir>/processed/S/` (no backend in the path: processing a second backend overwrites the first; `validate own` doesn't need processed files) |
| `process --clip NPZ` | the same for one KeypointSeq with no sessions row: session = the npz's folder name, clip = file name up to the first dot, condition inferred from the clip id. Writes `<clip>.strides.parquet` and `<clip>.windows.parquet` to `data/processed/<session>/` (`data/synthetic/processed/<session>/` for synthetic keypoints) unless `--out` |
| `overlay --clip NPZ [--video MP4] [--gif]` | skeleton + IC/TO markers + near-side knee and shin angles, on the video or on a plain canvas of `KeypointSeq.size`. Default output `<npz stem>.overlay.{mp4,gif}` |
| `label [--blind N] [--seed S] [--rater R] [--synthetic] [--root DIR] [--backend B] [--port P] [--no-browser]` | labeling app (§6) on `<root>/processed/*/windows.parquet` → `<root>/labels/labels.csv` (`--synthetic`: `data/synthetic/labels/hand_labels.csv`) |
| `baselines [--synthetic]` | fit and predict rules / LR / LightGBM. `--synthetic`: on `data/synthetic`, predictions to `data/synthetic/predictions/` |
| `build-sft [--synthetic]` | labels + windows → SFT JSONL (session split). `--synthetic`: `data/synthetic/sft/*.jsonl` |
| `train --model {1b,270m} [--sft-dir DIR] [--allow-synthetic] [--max-steps N] [--device D]` | LoRA fine-tune → `runs/<run_id>/` |
| `predict --model {1b,270m} [--synthetic] [--adapter RUN \| --zero-shot] [--consistency-runs R]` | logit grade + greedy tags + cue for the test windows (newest run of `--model` by default) → `predictions/<model>.csv`, `.meta.json`, `.consistency.json`. `--synthetic` only until real labels exist |
| `eval [--synthetic]` | all metrics → results/grader (`--synthetic`: results/synthetic/grader, never reported); `verifier` and `consistency` when an LLM's predictions are present |
| `validate own [--sessions CSV] [--backend B] [--other B2] [--session S]… [--synthetic]` | V5 directions, V6 stability ICC and, with `--other`, V7 backend agreement (`eval.own_validation`, EVALUATION §3) → results/validation. Recomputes strides and windows from `<CSV dir>/interim/<session>/<clip>.<backend>.npz` with the `process` code (`data.analysis`), so the backend is known; V7's matched stride pairs go to `<CSV dir>/processed/backend_pairs.<B>-<B2>.parquet` (sha256 in the result) |
| `validate {biocv,mocap}` | event and feature validation on public data → results/validation |
| `bench --trt [--synthetic] [--model {1b,270m}] [--adapter RUN \| --zero-shot]` | B2 TensorRT vs eager grade scoring pass on CUDA → `results/bench/trt.json` (`--synthetic`: synthetic SFT prompts and run → `results/synthetic/bench/`). Plain `bench` (B1 latency pipeline) is not written yet |
| `report` | regenerate the README results table |
| `synth --n-sessions K [--write-keypoints]` | synthetic sessions + rule labels for dry runs (`synthetic=True`). `--write-keypoints` also saves each clip as `data/synthetic/interim/<session>/<clip>.synthetic.npz` plus `data/synthetic/sessions.csv` (§4.6 schema + `synthetic` column) |
| `app` | coach demo app |

## 11. Testing and CI

- `tests/` mirrors `src/`. Synthetic gait (`data/synthetic.py`) is the main fixture: a 2D kinematic runner with **known** cadence, contact frames and per-stride geometry, plus a noise model (Gaussian jitter, dropouts, occasional left/right swaps).
- Markers: `gpu` (skipped in CI), `slow`, `data` (needs downloaded public data; skipped in CI).
- CI (`.github/workflows/ci.yml`): `uv sync --extra video --extra ml --extra viz --extra dev`, `ruff check`, `ruff format --check`, `pytest -m "not gpu and not data"`, plus `scripts/check_readme_numbers.py`.
- Each milestone's tests are listed in [TASKS.md](TASKS.md). A milestone is done when its tests pass and its result files exist.

## 12. Evidence base

| Decision | Source |
|---|---|
| ≥120 fps; 240 fps for contact time | [Reliability of 2D treadmill running analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6044590); [field reliability at 240 Hz](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6359244/) |
| 20-stride windows | [Riazati et al. 2019](https://researchportal.northumbria.ac.uk/en/publications/the-number-of-strides-required-for-treadmill-running-gait-analysi/) |
| 6+ min treadmill acclimation, fitted clothing, lateral view | [Video-based running analysis framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC4714754) |
| Strike-angle calibration and cut-offs | [Altman & Davis 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3278526) |
| Feature set; no loading-rate claims | [Wille et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4297621/) |
| Metronome condition and expected directions | [Heiderscheit et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3022995/) |
| Two pose backends, choose by validation | [Clinical HPE benchmark](https://www.nature.com/articles/s41598-025-22626-7); [lower-body model comparison](https://sjsp.aearedo.es/index.php/sjsp/article/download/pose-estimation-models-validated-lower-body-kinematics/219/3811); [VideoRun2D](https://arxiv.org/html/2608.19480v1) |
| Reference implementations | [Sports2D](https://github.com/davidpagnon/Sports2D); [Stenum et al. workflow](https://github.com/janstenum/GaitAnalysis-PoseEstimation) |
| Filter cutoffs (12 Hz events, 20 Hz features, to retune) | Literature practice of 6–8 Hz for marker kinematics ([Altman & Davis 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3278526); [OpenCap gait example](https://github.com/stanfordnmbl/opencap-processing/blob/main/Examples/example_gait_analysis.py)), adjusted per [ADR-0008](adr/0008-filter-cutoffs-events-vs-features.md) |
| Cue verifier | [FormCoach](https://arxiv.org/html/2508.07501v2) reports VLM-hallucinated feedback; [TAGS](https://openaccess.thecvf.com/content/ICCV2025W/AMFG/papers/Wang_Generating_Tennis_Action_Instruction_Based_on_a_Large_Language_Model_ICCVW_2025_paper.pdf) |
| Gemma 3 270M as a main comparison | [Introducing Gemma 3 270M](https://developers.googleblog.com/introducing-gemma-3-270m/) |
| TensorRT on the scoring pass only | [Torch-TensorRT HF LLM support list](https://docs.pytorch.org/TensorRT/tutorials/compile_hf_models.html); [TensorRT-LLM Windows deprecation](https://nvidia.github.io/TensorRT-LLM/0.21.0/overview.html) |
| Stand-in validation data | [BioCV](https://researchdata.bath.ac.uk/1258/); [AthletePose3D](https://arxiv.org/html/2503.07499v2); [Guo et al. 2026](https://zenodo.org/records/18399031); [Running Injury Clinic](https://figshare.com/articles/dataset/Running_Injury_Clinic_Kinematic_Dataset/24255795); [Fukuchi 2017](https://peerj.com/articles/3298); [CMU MoCap](https://paperswithcode.com/dataset/cmu-motion-capture) |
