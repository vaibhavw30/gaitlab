# ADR-0001: The LLM sees feature summaries, not keypoints

- Status: accepted
- Date: 2026-11-03

## Context
Raw keypoint sequences are long (frames × joints × 2) and noisy. Measurements such as cadence or shin angle at contact have standard, testable definitions (Wille et al. 2014; Altman & Davis 2012). Published coaching systems that let a VLM look at frames report hallucinated feedback (FormCoach, 2025).

## Decision
Pose → deterministic per-stride features → 20-stride window summaries (median, IQR). The LLM gets a short, fixed-format text prompt of those summaries. It never sees pixels or keypoints.

## Consequences
- Measurement correctness is unit-testable on synthetic and mocap data, independent of the LLM.
- The LLM is small (1B / 270M) and its input is short, so the scoring pass is cheap and has a static shape.
- Strong tabular baselines (LR, LightGBM) on the same features are mandatory; the LLM has to justify itself against them.
