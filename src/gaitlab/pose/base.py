"""Pose backend interface. Every backend returns a canonical ``KeypointSeq``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from gaitlab.types import KeypointSeq


class PoseBackend(Protocol):
    """Runs a pose model over decoded frames.

    Implementations: ``mediapipe_backend.MediaPipeBackend``, ``rtmpose_backend.RTMPoseBackend``.
    """

    name: str

    def run(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        fps: float,
        size: tuple[int, int],
    ) -> KeypointSeq:
        """Estimate keypoints for every frame.

        Parameters
        ----------
        frames : iterable of (t_seconds, RGB uint8 array of shape (H, W, 3)), in order.
        fps : nominal fps of the clip.
        size : (width, height).

        Returns
        -------
        KeypointSeq in the canonical skeleton, pixels, y down. Frames with no detection get
        NaN xy and vis 0. Single-person: keep the most confident / largest person.
        """
        ...


def get_backend(name: str, **kwargs: object) -> PoseBackend:
    """Factory: ``"mediapipe"`` or ``"rtmpose"``. Imports lazily so CPU tests don't need them."""
    if name == "mediapipe":
        from gaitlab.pose.mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend(**kwargs)  # type: ignore[arg-type]
    if name == "rtmpose":
        from gaitlab.pose.rtmpose_backend import RTMPoseBackend

        return RTMPoseBackend(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown pose backend {name!r}")
