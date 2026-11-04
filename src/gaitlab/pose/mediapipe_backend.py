"""MediaPipe Pose Landmarker backend (TASKS M0, FR-2).

Tasks API ``PoseLandmarker`` in VIDEO running mode, one pose, heavy model by default
(``configs/default.yaml: pose.mediapipe_model``, fetched by ``scripts/download_models.sh``).

- Each frame goes in as ``mp.Image(SRGB)`` with an integer ms timestamp from its PTS
  (``video_timestamps_ms``; VIDEO mode rejects non-increasing timestamps).
- ``pose_landmarks[0]`` gives normalized x, y and ``visibility``; x · width, y · height → px,
  then 33 → canonical with ``skeleton.to_canonical(..., "mediapipe33")``.
- No detection → NaN xy and vis 0 for every joint.
- ``meta = {"mediapipe": version, "model": file name, "model_sha256": sha of the .task file}``.

mediapipe is imported only inside ``tasks_detector``, so this module imports on CPU-only
installs without the ``mediapipe`` extra.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import numpy as np

from gaitlab.pose.skeleton import to_canonical
from gaitlab.types import KeypointSeq

N_LANDMARKS = 33

Landmarks = tuple[np.ndarray, np.ndarray]
"""(33, 2) normalized x, y (fractions of width, height) and (33,) visibility in [0, 1]."""
DetectFn = Callable[[np.ndarray, int], Landmarks | None]
"""(RGB uint8 (H, W, 3), timestamp ms) → landmarks, or None if no person."""
OpenDetector = Callable[[str], AbstractContextManager[tuple[DetectFn, dict]]]
"""model path → context manager yielding (detect function, backend meta)."""


def next_timestamp_ms(t: float, last_ms: int | None) -> int:
    """``round(t · 1000)`` ms, bumped to ``last_ms + 1`` if it would not exceed ``last_ms``."""
    ms = round(float(t) * 1000.0)
    return ms if last_ms is None or ms > last_ms else last_ms + 1


def video_timestamps_ms(t: np.ndarray) -> list[int]:
    """Integer ms timestamps for VIDEO mode, strictly increasing.

    Parameters
    ----------
    t : (T,) float
        Frame times in s from PTS (``io.video.iter_frames``).

    Returns
    -------
    list of int
        ``round(t · 1000)``, bumped by 1 ms wherever that would not exceed the previous value
        (sub-ms spacing or duplicate PTS). Bumps shift a frame's timestamp by at most a few ms
        and only affect MediaPipe's tracking; ``KeypointSeq.t`` keeps the true PTS times.
    """
    out: list[int] = []
    for ti in np.asarray(t, dtype=np.float64):
        out.append(next_timestamp_ms(ti, out[-1] if out else None))
    return out


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def tasks_detector(model_path: str) -> Iterator[tuple[DetectFn, dict]]:
    """Open a MediaPipe Tasks ``PoseLandmarker`` (VIDEO mode, one pose) for one clip."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} not found; run scripts/download_models.sh")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    meta = {
        "mediapipe": mp.__version__,
        "model": Path(model_path).name,
        "model_sha256": file_sha256(model_path),
    }
    with PoseLandmarker.create_from_options(opts) as landmarker:

        def detect(rgb: np.ndarray, ts_ms: int) -> Landmarks | None:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(image, ts_ms)
            if not result.pose_landmarks:
                return None
            lms = result.pose_landmarks[0]
            xy = np.array([[p.x, p.y] for p in lms], dtype=np.float32)
            vis = np.array(
                [p.visibility if p.visibility is not None else 0.0 for p in lms],
                dtype=np.float32,
            )
            return xy, vis

        yield detect, meta


class MediaPipeBackend:
    """``PoseBackend`` running MediaPipe Pose Landmarker.

    Parameters
    ----------
    model_path : str
        Path to the ``.task`` model (heavy by default).
    open_detector : callable, optional
        Seam for tests: model path → context manager yielding ``(detect, meta)``. Defaults to
        ``tasks_detector``. A fresh detector is opened per ``run`` because VIDEO mode keeps
        tracking state and needs increasing timestamps within one clip.
    """

    name = "mediapipe"

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_heavy.task",
        open_detector: OpenDetector | None = None,
    ) -> None:
        self.model_path = model_path
        self._open_detector = open_detector or tasks_detector

    def run(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        fps: float,
        size: tuple[int, int],
    ) -> KeypointSeq:
        """Estimate keypoints for every frame; see ``pose.base.PoseBackend.run``."""
        width, height = size
        ts: list[float] = []
        xy_rows: list[np.ndarray] = []
        vis_rows: list[np.ndarray] = []
        last_ms: int | None = None
        with self._open_detector(self.model_path) as (detect, meta):
            for t, rgb in frames:
                if rgb.shape[:2] != (height, width):
                    raise ValueError(
                        f"frame size {rgb.shape[1]}x{rgb.shape[0]} != clip size {width}x{height}"
                    )
                last_ms = next_timestamp_ms(t, last_ms)
                found = detect(rgb, last_ms)
                if found is None:
                    xy = np.full((N_LANDMARKS, 2), np.nan, dtype=np.float32)
                    vis = np.zeros(N_LANDMARKS, dtype=np.float32)
                else:
                    xy, vis = found
                    xy = xy * np.array([width, height], dtype=np.float32)
                    vis = np.clip(vis, 0.0, 1.0)
                ts.append(float(t))
                xy_rows.append(xy)
                vis_rows.append(vis)
        if not ts:
            raise ValueError("no frames to run pose estimation on")
        cxy, cvis = to_canonical(np.stack(xy_rows), np.stack(vis_rows), "mediapipe33")
        return KeypointSeq(
            xy=cxy,
            vis=cvis,
            t=np.asarray(ts, dtype=np.float64),
            fps=float(fps),
            size=(int(width), int(height)),
            backend=self.name,
            meta=dict(meta),
        )
