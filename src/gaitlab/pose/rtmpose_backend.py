"""RTMPose backend via rtmlib (TASKS M0). Second backend; primary chosen by ADR-0005.

Implementation notes for Claude Code:
- ``from rtmlib import Body`` (or ``BodyWithFeet``/a custom ``RTMPose`` with a Halpe-26 body7
  model, so heel and big toe exist). Check the installed rtmlib's model zoo and pick a
  26-keypoint model; COCO-17 models have no feet and must not be used here.
- ``backend="onnxruntime", device="cuda"``; ``mode`` from config ``pose.rtmpose_mode``.
- Calls return ``keypoints (N, K, 2)`` in pixels and ``scores (N, K)``. Keep the person with the
  highest mean score (single runner). Map with ``skeleton.to_canonical(..., "halpe26")``.
- rtmlib expects BGR (OpenCV convention): convert RGB → BGR with ``rgb[..., ::-1]``.
- Scores are not calibrated visibilities. Record the score distribution in ``meta`` so
  ``clean.visibility_min`` can be tuned per backend if needed.
- For angle cross-checks, Sports2D (same RTMPose models) can be run on the same clip;
  see tests marked ``data``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from gaitlab.types import KeypointSeq


class RTMPoseBackend:
    name = "rtmpose"

    def __init__(self, mode: str = "performance", device: str = "cuda") -> None:
        self.mode = mode
        self.device = device

    def run(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        fps: float,
        size: tuple[int, int],
    ) -> KeypointSeq:
        raise NotImplementedError("TASKS.md M0: RTMPose backend")
