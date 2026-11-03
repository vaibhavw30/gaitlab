"""Video decoding with PyAV (TASKS M0, FR-1).

Contract:
- ``probe(path) -> VideoInfo`` with fps (average_rate), size, n_frames (may be estimated),
  duration_s, codec, and ``vfr`` (True if PTS deltas vary by more than 1%).
- ``iter_frames(path) -> Iterator[tuple[float, np.ndarray]]`` yielding (t_seconds from PTS ×
  time_base, RGB uint8 (H, W, 3)) in presentation order. Use ``frame.to_ndarray(format="rgb24")``.
  Handle rotation metadata (phones record portrait with a rotate tag): apply it so the runner
  is upright, and set ``VideoInfo.size`` after rotation.
- ``check_fps(info, cfg.capture)`` raises ``LowFpsError`` below ``min_fps`` unless allowed.
Test: generate a tiny clip in-test with PyAV (e.g. 64×48, 30 frames at 120 fps) and check
timestamps, count and shape.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from gaitlab.config import CaptureCfg


class LowFpsError(ValueError):
    pass


@dataclass
class VideoInfo:
    path: str
    fps: float
    size: tuple[int, int]
    n_frames: int
    duration_s: float
    codec: str
    vfr: bool


def probe(path: str) -> VideoInfo:
    raise NotImplementedError("TASKS.md M0: video decode")


def iter_frames(path: str) -> Iterator[tuple[float, np.ndarray]]:
    raise NotImplementedError("TASKS.md M0: video decode")


def check_fps(info: VideoInfo, cfg: CaptureCfg, allow_low_fps: bool = False) -> None:
    if info.fps < cfg.min_fps and not allow_low_fps:
        raise LowFpsError(
            f"{info.path}: {info.fps:.1f} fps < {cfg.min_fps} (use --allow-low-fps for smoke tests)"
        )
