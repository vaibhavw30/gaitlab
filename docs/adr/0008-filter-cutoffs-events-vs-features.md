# ADR-0008: Separate filter cutoffs for event detection and features

- Status: accepted (starting values; retune on BioCV in M0)
- Date: 2026-11-03

## Context
Running studies commonly low-pass marker data at 6–8 Hz (Altman & Davis 2012 used 8 Hz; OpenCap's walking example uses 6 Hz). While scaffolding, the synthetic runner (`data/synthetic.py`, 120 fps) showed three things:
- An 8 Hz zero-phase filter on keypoints shifted detected initial contact one frame late, systematically. The foot reverses sharply at contact, and the filter rounds that corner. Overstride and foot-strike angle measured at that frame are biased as a result.
- With no feature filter, jitter inflates range-based features (vertical oscillation is a max − min).
- Knee flexion and shin angle at contact change several degrees per frame around contact at 120 fps, so they are sensitive to one-frame event errors. That supports the 240 fps capture for contact-instant metrics (ADR-0006).

These are synthetic observations used to pick defaults. They are not results and are not reported.

## Decision
- `clean.lowpass_hz` (event detection) starts at **12 Hz**.
- `clean.feature_lowpass_hz` (signals used for per-stride features) starts at **20 Hz**.
- Both are retuned in M0 against BioCV force-plate contacts and mocap angles (EVALUATION V1/V2), and this ADR is updated with the chosen values and the result file that justified them.

## Consequences
- `signal.clean.clean_keypoints(..., for_features=True)` selects the feature cutoff; `gait.features.analyze_clip` cleans twice.
- Tests compare features computed at the *true* events (isolating feature math) and end-to-end (events + features), with looser end-to-end tolerances for knee and shin angles.
