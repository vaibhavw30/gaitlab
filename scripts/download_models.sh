#!/usr/bin/env bash
# Downloads the MediaPipe Pose Landmarker model used by pose/mediapipe_backend.py.
# URL pattern from the MediaPipe pose landmarker docs; if it 404s, take the current
# link from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
set -euo pipefail
mkdir -p models
URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
curl -fL "$URL" -o models/pose_landmarker_heavy.task
# sha256sum on Linux, shasum on macOS
if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
$SHA models/pose_landmarker_heavy.task | tee models/pose_landmarker_heavy.task.sha256
echo "Saved models/pose_landmarker_heavy.task"
# RTMPose (rtmlib) downloads its ONNX models on first use into its own cache.
# Gemma weights come from Hugging Face on first `gaitlab train` (accept the license first).
