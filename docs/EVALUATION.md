# Evaluation

Every number a reader sees comes from here, through `report/results.py`. If it isn't in `results/*.json` with `synthetic: false` and `git_dirty: false`, it doesn't exist.

## 1. Questions → metrics

| # | Question | Metric | Data | Result file |
|---|---|---|---|---|
| V1 | Are contacts detected at the right time? | IC and TO error (frames, ms): median, IQR, % within ±1 frame; missed and extra events | BioCV running trials vs force plates | `results/validation/biocv_events.json` |
| V2 | Are angles right? | Knee flexion at IC, shin angle, trunk lean: bias, 95% limits of agreement (Bland–Altman), RMSE vs mocap | BioCV | `results/validation/biocv_angles.json` |
| V3 | Is the feature code right? | Absolute error vs exact values, per feature | Projected mocap (noise-free and noisy) | `results/validation/mocap_features.json` |
| V4 | Is cadence right on my video? | MAE (steps/min) vs hand counts | Own sessions | `results/validation/own_cadence.json` |
| V5 | Do features move the right way? | Sign and size of condition − natural differences with bootstrap CI: metronome → cadence ↑, overstride ↓, vert_osc ↓; overstride → overstride_L ↑, shin ↑ | Own sessions | `results/validation/directions.json` |
| V6 | Are features stable? | ICC(2,1) across windows of the same condition and session | Own sessions | `results/validation/stability.json` |
| V7 | Do two pose models agree? | Bland–Altman per feature, MediaPipe vs RTMPose | Own + BioCV | `results/validation/backend_agreement.json` |
| G1 | Does the grader agree with me? | Exact, within-1, **quadratic-weighted kappa (QWK)**, each with a 95% CI | Test sessions | `results/grader/ladder.json` |
| G2 | What's the ceiling? | Intra-rater QWK and exact agreement on the blind re-grade | 40 windows | `results/grader/intra_rater.json` |
| G3 | Are tags right? | Macro-F1 over the tag vocabulary; per-tag F1 | Test sessions | `results/grader/tags.json` |
| G4 | Are cues grounded? | Cue-verifier pass rate; failure reasons histogram | Test sessions | `results/grader/verifier.json` |
| G5 | Are probabilities honest? | ECE (10 equal-width bins on max P), Brier (multiclass) | Test sessions | `results/grader/calibration.json` |
| G6 | Is it consistent? | Identical-grade rate over 5 runs (greedy; and sampling at T=0.7 for cues) | Test sessions | `results/grader/consistency.json` |
| B1 | How fast is each stage? | p50, p95, mean (ms) per stage; end-to-end fps; peak VRAM | Fixed 60-s clip | `results/bench/stages.json` |
| B2 | What does TensorRT buy? | Speedup vs eager FP16 at batch 1 and 16, parity (argmax identical %, max \|ΔP\|) | Test prompts | `results/bench/trt.json` |

## 2. Grader ladder

Same split and same windows for every model: rules, logistic regression, LightGBM, Gemma 3 1B zero-shot, Gemma 3 1B + LoRA, Gemma 3 270M + LoRA. The ladder chart shows QWK with 95% CI per model and a dashed line at the intra-rater QWK.

## 3. Protocol details

- **Bootstrap:** 2,000 resamples. Resample **sessions** (cluster bootstrap), then the windows within them; percentile CIs. Seed fixed in config.
- **QWK:** `sklearn.metrics.cohen_kappa_score(weights="quadratic")` with labels fixed to [1..5]; tested against a hand-computed example.
- **ECE:** confidence = max P(grade), 10 equal-width bins, weighted by bin count.
- **Within-1:** |pred − label| ≤ 1.
- **Multi-label tags:** macro-F1 over tags that appear at least once in test labels; the rest are listed, not averaged.
- **No test peeking:** thresholds (features.yaml), LoRA settings and model choice are fixed on train + dev. The test split is scored once per final model; any rerun after a code change must say so in the result's `notes`.
- **Synthetic runs** compute the same metrics into `results/synthetic/` to test plumbing. They're never reported.

## 4. Bench protocol

- Warmup 20 iterations, measure 200 (config). CUDA events around each stage, `torch.cuda.synchronize()` before reading.
- Report the GPU name, driver, CUDA, cuDNN, TensorRT, torch and torch-tensorrt versions, and power mode (laptop on AC, performance profile). Record those in the result.
- One Nsight Systems trace (`nsys profile`) of the full pipeline on the fixed clip, saved under `results/bench/` as a screenshot plus the summary table.
- The TensorRT speedup is reported only if parity is 100% argmax agreement on the test prompts.

## 5. What goes in the README

`report/readme_table.py` renders a table from `results/` only: V1, V4, G1 (+ G2 ceiling), G4, B2. Each cell links to its JSON. CI (`scripts/check_readme_numbers.py`) fails if a digit-bearing claim appears outside the generated block, except in an allowlist (dates, version numbers, citations).
