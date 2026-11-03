# ADR-0006: Capture at ≥120 fps, warm up 6+ minutes, label 20-stride windows

- Status: accepted
- Date: 2026-11-03

## Context
- Most running-video researchers treat 120 fps as the minimum; reliability studies used 240 Hz.
- Treadmill kinematics change during roughly the first 6 minutes; 6–10 minutes of acclimation is recommended.
- 20 consecutive strides are needed for stable treadmill kinematics (Riazati et al. 2019).
- Foot-strike angle is defined relative to the standing foot angle (Altman & Davis 2012).

## Decision
120 fps minimum (240 for contact-time sessions); 6+ minute off-camera warm-up; a 3-s standing clip per session; windows of 20 strides.

## Consequences
Fewer windows per minute of video (about 4 per minute). The labeling target of ~200 windows needs ~50 minutes of usable footage.
