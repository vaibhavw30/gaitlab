"""MediaPipe Pose Landmarker backend (TASKS M0).

Implementation notes for Claude Code:
- Use the Tasks API: ``mediapipe.tasks.python.vision.PoseLandmarker`` with
  ``running_mode=VisionTaskRunningMode.VIDEO`` and ``num_poses=1``; model path from
  ``configs/default.yaml: pose.mediapipe_model`` (downloaded by scripts/download_models.sh).
- Feed ``mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)`` with a strictly increasing
  integer timestamp in ms derived from the frame's PTS (``round(t * 1000)``; bump by 1 on ties).
- ``pose_landmarks[0]`` gives normalized x, y (0..1) and ``visibility``. Convert to pixels:
  ``x * width``, ``y * height``. Map 33 → canonical with
  ``skeleton.to_canonical(..., "mediapipe33")``.
- No detection in a frame → NaN xy, vis 0 for all joints.
- Record ``meta = {"mediapipe": mediapipe.__version__, "model_sha256": <sha of .task file>}``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from gaitlab.types import KeypointSeq


class MediaPipeBackend:
    name = "mediapipe"

    def __init__(self, model_path: str = "models/pose_landmarker_heavy.task") -> None:
        self.model_path = model_path

    def run(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        fps: float,
        size: tuple[int, int],
    ) -> KeypointSeq:
        raise NotImplementedError("TASKS.md M0: MediaPipe backend")
