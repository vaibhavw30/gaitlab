# GaitLab — Product Requirements Document

| | |
|---|---|
| Owner | Vaibhav Wudaru |
| Status | Draft v1 — 2026-11-03 |
| Companion docs | [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md) · [DATA.md](DATA.md) · [EVALUATION.md](EVALUATION.md) · [TASKS.md](TASKS.md) · [adr/](adr/) |

## 1. Summary

GaitLab turns a phone video of someone running on a treadmill into a per-window running-form grade (1–5), issue tags, and one plain-language coaching cue. The pipeline is: pose keypoints → deterministic per-stride biomechanics → a small fine-tuned LLM that grades and coaches. Every reported number comes from an evaluation script that writes `results/*.json`. Nothing else is a source for a number.

## 2. Problem

Recreational runners rarely get form feedback. Lab gait analysis needs motion capture. Phone apps either show raw angles nobody can interpret or give generic advice that isn't tied to measurements. Research shows that a handful of side-view (sagittal) measurements track meaningful loading. Examples are cadence, how far the foot lands ahead of the body, shin and knee angle at contact, and vertical bounce. These can be measured from a single 2D video (see Evidence in [TECHNICAL_PLAN.md §12](TECHNICAL_PLAN.md#12-evidence-base)).

## 3. Users

| User | Need |
|---|---|
| **Primary: the author as runner** | Upload a session, see which windows are graded poorly and why, get one cue at a time, and compare sessions over weeks. |
| **Secondary: technical reviewers** (interviewers, open-source readers) | Understand the system in five minutes from the README. Reproduce the evaluation. Trust that the numbers are measured. |

Single runner, single rater. That limitation is stated, not hidden.

## 4. Goals

1. **G1 Measure:** extract per-stride sagittal gait features from 2D video, validated against public ground truth (BioCV force plates and motion capture) and hand counts.
2. **G2 Grade:** produce a 1–5 grade per 20-stride window that agrees with the author's labels on held-out sessions. Agreement is reported next to the author's own re-grade agreement, which is the ceiling.
3. **G3 Coach:** produce issue tags and one cue per window. Every cue is grounded in a feature that is actually out of range, which a rule-based verifier checks.
4. **G4 Run fast on a laptop GPU:** profile every stage and accelerate the GPU hot spots with TensorRT, keeping output parity with eager PyTorch.
5. **G5 Be honest and reproducible:** one command regenerates every reported number, each result file carries provenance, and synthetic runs are flagged and never reported.

## 5. Non-goals (v1)

- Medical diagnosis, injury prediction, or any claim about ground-reaction loading *rates*. Wille et al. 2014 found sagittal kinematics do not estimate vertical loading rate (R² 0.04).
- Real-time feedback during a run.
- Multi-person scenes, overground running, or outdoor footage.
- Frontal-plane metrics (pelvic drop, knee valgus, crossover). A rear view is optional and out of scope for v1 grading.
- A mobile app. The deliverable is a CLI plus a local Gradio app.
- Generalizing to other runners. v1 is personalized: one runner, one rater.

## 6. User stories

| ID | Story | Acceptance |
|---|---|---|
| US-1 | As the runner, I drop a session video and get an overlay video with skeleton, contact markers and live knee and shin angles. | `gaitlab overlay` produces an mp4 or GIF; contacts render on the right frames for a BioCV clip. |
| US-2 | As the runner, I see a per-window table of features and a grade with probabilities. | The coach app shows windows, feature medians and IQR, grade, P(grade) and tags. |
| US-3 | As the runner, I get one cue per window that cites the metric behind it. | Every cue shown has passed the verifier; a failed cue is suppressed and logged. |
| US-4 | As the runner, I compare two sessions. | `gaitlab compare` outputs feature deltas with the run-to-run spread. |
| US-5 | As the labeler, I grade windows quickly with the video, overlay and features side by side. | The label app writes `labels.csv` rows matching the schema in DATA.md and supports blind re-grade mode. |
| US-6 | As a reviewer, I reproduce every number. | `gaitlab validate`, `gaitlab eval`, `gaitlab bench`, then `gaitlab report` regenerate `results/*.json` and the README table; CI checks that no hard-coded numbers sit in the README. |
| US-7 | As a reviewer, I see what TensorRT changed. | The bench report shows per-stage p50 and p95, eager vs TensorRT, grade parity, and hardware and software versions. |

## 7. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Decode video with frame-accurate timestamps (PyAV). Reject clips under 100 fps unless `--allow-low-fps` is set; such clips are excluded from contact-time metrics. | Must |
| FR-2 | Two pose backends behind one interface: MediaPipe Pose Landmarker (heavy) and RTMPose (Halpe-26 via rtmlib/Sports2D). Both output one canonical skeleton that includes heel and toe. | Must |
| FR-3 | Signal cleaning: visibility gating, short-gap interpolation, zero-phase Butterworth low-pass with separate cutoffs for event detection and features (ADR-0008; configurable, retuned on BioCV). | Must |
| FR-4 | Gait events: initial contact and toe-off per foot, from foot kinematics. | Must |
| FR-5 | Per-stride features: cadence, overstride, shin angle at contact, knee flexion at contact, peak stance knee flexion, foot strike angle (calibrated by a standing clip), contact time, duty factor, vertical oscillation, trunk lean. All lengths are normalized by leg length. | Must |
| FR-6 | Windowing: 20-stride windows with median and IQR per feature, plus a quality score (fraction of usable frames). | Must |
| FR-7 | Labeling app with normal and blind re-grade modes. | Must |
| FR-8 | Baselines: rules, logistic regression, LightGBM. | Must |
| FR-9 | LLM grader: Gemma 3 1B IT and Gemma 3 270M IT, each with a LoRA adapter. The grade comes from next-token probabilities over "1"–"5"; tags and the cue are generated text. | Must |
| FR-10 | Cue verifier: a cue must reference at least one feature outside its configured range for that window. | Must |
| FR-11 | Evaluation: metrics per [EVALUATION.md](EVALUATION.md), on a session-level split, with bootstrap CIs. | Must |
| FR-12 | Benchmarks: per-stage latency, end-to-end throughput, peak VRAM, and eager vs Torch-TensorRT for the scoring pass with parity. | Must |
| FR-13 | Results provenance: git SHA, config hash, data manifest hash, synthetic flag, hardware, library versions. | Must |
| FR-14 | Validation on public data: BioCV contact events and angles; projected mocap for exact feature checks. | Must |
| FR-15 | RTMPose TensorRT FP16 engine and its benchmark. | Should |
| FR-16 | Population percentiles from the Running Injury Clinic dataset. | Could |
| FR-17 | MCP tools (`analyze_session`, `compare_sessions`, `get_norms`) and a small coach agent. | Could |

## 8. Non-functional requirements

- **Reproducibility:** seeds fixed; one config file per run, hashed into results.
- **Honesty:** no number appears in README, docs or the app that isn't read from `results/*.json`. Synthetic results are written to `results/synthetic/` and are excluded from the README table.
- **Privacy:** raw videos never leave the machine; `data/` is git-ignored. Public datasets are downloaded by the user under their own terms and are never redistributed.
- **Portability:** Linux or WSL2 with an NVIDIA GPU for training and benchmarks. Every test except those marked `gpu` runs on CPU in CI.
- **Code quality:** typed Python 3.11, ruff, pytest. Pure functions in `gait/` and `eval/` with no I/O.

## 9. Success measures

These are measurements to report, not targets to hit. The PRD sets no thresholds, because a threshold invites tuning toward it.

| Area | Report |
|---|---|
| Event detection | Initial-contact error vs BioCV force plates (frames and ms); cadence MAE vs hand counts |
| Feature validity | Feature error on projected mocap; direction checks across conditions (metronome raises cadence and lowers overstride and bounce) |
| Grader | Exact and within-1 agreement, quadratic-weighted kappa with 95% CI, the author's intra-rater kappa as the ceiling, ladder comparison vs baselines |
| Coaching | Tag macro-F1; cue-verifier pass rate |
| Calibration | ECE and Brier on grade probabilities |
| Consistency | Identical-grade rate over 5 runs |
| Speed | Per-stage p50 and p95, end-to-end fps, peak VRAM, TensorRT speedup with 100% grade parity |

## 10. Milestones

| M | Name | Outcome |
|---|---|---|
| M0 | Stand-in data | Pipeline runs on Mixkit, Sports2D demo, BioCV and projected mocap; synthetic labels drive training and eval end to end |
| M1 | Capture and signal | Own sessions recorded; pose → events → cadence validated |
| M2 | Features and labels | All features, labeling app, ~200 labeled windows, baselines |
| M3 | Grader | LoRA 1B and 270M, full eval, intra-rater ceiling, ladder chart |
| M4 | Speed and polish | TensorRT scoring pass, Nsight trace, README from results |
| M5 | Stretch | RTMPose TensorRT engine, second validation set, MCP agent |

Detailed tasks and acceptance criteria: [TASKS.md](TASKS.md).

## 11. Risks

| Risk | Mitigation |
|---|---|
| Labels all cluster at one grade | Record deliberate bad-form conditions (DATA.md capture protocol) |
| LightGBM matches the LLM on the grade | Report it. The LLM's value is tags and cues; the 270M vs 1B comparison is still informative |
| Far-side limb occluded in side view | Near-side features only; asymmetry marked low-confidence |
| Pose jitter at contact | 120 fps minimum, 240 fps for contact-time sessions; validate on BioCV |
| Gemma not supported by Torch-TensorRT's LLM generation tools | Compile only the fixed-shape scoring forward pass ([ADR-0004](adr/0004-tensorrt-placement.md)) |
| Dataset terms restrict use | Read each license before download; never commit or redistribute |
| A placeholder number leaks into a resume | `/check-claims` command and a CI check that the README has no numbers outside the generated table |

## 12. Open questions

- Laptop GPU model and VRAM; Linux or WSL2?
- Do BioCV and AthletePose3D terms allow this use? (Read before M0 downloads.)
- Will the Ole Miss stride-level dataset be shared?
- Interview date, which sets how much of M4 and M5 is realistic.
