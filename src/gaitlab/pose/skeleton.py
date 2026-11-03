"""Canonical 13-joint skeleton and maps from each pose backend (TECHNICAL_PLAN §4.1)."""

from __future__ import annotations

import numpy as np

CANONICAL: tuple[str, ...] = (
    "nose",
    "l_shoulder",
    "r_shoulder",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
    "l_heel",
    "r_heel",
    "l_toe",
    "r_toe",
)
J: dict[str, int] = {name: i for i, name in enumerate(CANONICAL)}
"""Name → index. Always index joints by name: ``xy[:, J["l_heel"]]``."""

# MediaPipe Pose Landmarker (33 landmarks). foot_index is the toe.
MEDIAPIPE_33: dict[str, int] = {
    "nose": 0,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_hip": 23,
    "r_hip": 24,
    "l_knee": 25,
    "r_knee": 26,
    "l_ankle": 27,
    "r_ankle": 28,
    "l_heel": 29,
    "r_heel": 30,
    "l_toe": 31,
    "r_toe": 32,
}

# Halpe-26 (RTMPose body7 26-keypoint models). Toe = big toe.
HALPE_26: dict[str, int] = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
    "l_toe": 20,
    "r_toe": 21,
    "l_heel": 24,
    "r_heel": 25,
}

# COCO-17 has no heel or toe; those joints become NaN.
COCO_17: dict[str, int] = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

SOURCE_MAPS: dict[str, dict[str, int]] = {
    "mediapipe33": MEDIAPIPE_33,
    "halpe26": HALPE_26,
    "coco17": COCO_17,
}


def to_canonical(
    xy: np.ndarray, vis: np.ndarray, source: str
) -> tuple[np.ndarray, np.ndarray]:
    """Map backend keypoints to the canonical skeleton.

    Parameters
    ----------
    xy : (T, K, 2) source keypoints, pixels.
    vis : (T, K) source visibility or confidence.
    source : key of ``SOURCE_MAPS``.

    Returns
    -------
    (T, 13, 2) xy and (T, 13) vis. Joints the source lacks are NaN with vis 0.
    """
    mapping = SOURCE_MAPS[source]
    n_t = xy.shape[0]
    out_xy = np.full((n_t, len(CANONICAL), 2), np.nan, dtype=np.float32)
    out_vis = np.zeros((n_t, len(CANONICAL)), dtype=np.float32)
    for name, src_idx in mapping.items():
        out_xy[:, J[name]] = xy[:, src_idx]
        out_vis[:, J[name]] = vis[:, src_idx]
    return out_xy, out_vis


def side_joints(side: str) -> dict[str, int]:
    """Indices of one side's shoulder, hip, knee, ankle, heel, toe. ``side`` is "l" or "r"."""
    if side not in ("l", "r"):
        raise ValueError(f"side must be 'l' or 'r', got {side!r}")
    return {
        part: J[f"{side}_{part}"] for part in ("shoulder", "hip", "knee", "ankle", "heel", "toe")
    }
