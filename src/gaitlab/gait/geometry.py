"""Coordinate conventions and small geometric helpers. Pure functions.

Conventions: arrays are pixels. Stored data is y-down; call ``to_y_up`` once. Angles in degrees.
``d`` is the facing direction in the image: +1 = facing right (+x), -1 = facing left.
"""

from __future__ import annotations

import numpy as np

from gaitlab.pose.skeleton import J, side_joints


def to_y_up(xy: np.ndarray, height: int) -> np.ndarray:
    """Convert y-down pixel coordinates to y-up (origin bottom-left). Works on any (..., 2)."""
    out = np.array(xy, dtype=np.float64, copy=True)
    out[..., 1] = height - out[..., 1]
    return out


def near_side(vis: np.ndarray) -> str:
    """Side ("l"/"r") with higher mean visibility over hip, knee, ankle, heel, toe."""
    parts = ("hip", "knee", "ankle", "heel", "toe")
    score = {}
    for side in ("l", "r"):
        idx = [J[f"{side}_{p}"] for p in parts]
        score[side] = float(np.nanmean(vis[:, idx]))
    return "l" if score["l"] >= score["r"] else "r"


def facing_direction(xy_up: np.ndarray, side: str) -> int:
    """+1 if toes point to +x (runner faces right), else -1. Median over valid frames."""
    sj = side_joints(side)
    dx = xy_up[:, sj["toe"], 0] - xy_up[:, sj["heel"], 0]
    med = np.nanmedian(dx)
    if not np.isfinite(med):
        raise ValueError("cannot determine facing direction: no valid heel/toe frames")
    return 1 if med >= 0 else -1


def leg_length(xy_up: np.ndarray, side: str) -> float:
    """Median over frames of |hip-knee| + |knee-ankle| (pixels). Robust to knee bend."""
    sj = side_joints(side)
    hip, knee, ankle = xy_up[:, sj["hip"]], xy_up[:, sj["knee"]], xy_up[:, sj["ankle"]]
    seg = np.linalg.norm(hip - knee, axis=-1) + np.linalg.norm(knee - ankle, axis=-1)
    val = float(np.nanmedian(seg))
    if not np.isfinite(val) or val <= 0:
        raise ValueError("cannot compute leg length")
    return val


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle ABC in degrees for (..., 2) arrays. 180 = straight."""
    v1 = a - b
    v2 = c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def signed_angle_from_vertical_deg(
    lower: np.ndarray, upper: np.ndarray, d: int
) -> np.ndarray:
    """Angle of the segment lower→upper from vertical (y-up); positive when ``upper`` is ahead.

    Used for trunk lean (hip→shoulder): positive = leaning forward.
    """
    dx = d * (upper[..., 0] - lower[..., 0])
    dy = upper[..., 1] - lower[..., 1]
    return np.degrees(np.arctan2(dx, dy))
