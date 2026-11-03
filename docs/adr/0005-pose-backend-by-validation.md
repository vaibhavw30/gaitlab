# ADR-0005: Choose the primary pose backend by validation

- Status: proposed (decision pending M0 results)
- Date: 2026-11-03

## Context
Published comparisons disagree. A clinical benchmark (Sci Rep 2025) found the RTMPose family most accurate among 2D estimators. A lower-body comparison found MediaPipe ahead of YOLO and MoveNet on knee angle while jogging. A sprint study put the best knee RMSE around 5–6°.

## Decision
Run MediaPipe Pose Landmarker (heavy) and RTMPose (Halpe-26) behind one interface. Choose the primary by V1/V2 on BioCV (results/validation/biocv_*.json). Fill in the outcome here.

## Consequences
Two extraction runs per clip during M0–M2. The README names only the backend that ships.
