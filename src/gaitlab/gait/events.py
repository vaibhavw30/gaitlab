"""Initial contact (IC) and toe-off (TO) detection from foot kinematics (TECHNICAL_PLAN §5.2).

Treadmill assumption: the runner stays roughly in place, so a stance foot moves **backward**
with the belt while it's near ground level. Validate against BioCV force plates (EVALUATION V1)
before trusting on real video; compare with Stenum et al.'s method before changing this one.
"""

from __future__ import annotations

import numpy as np

from gaitlab.config import EventsCfg
from gaitlab.pose.skeleton import side_joints
from gaitlab.types import Events


def contact_mask(
    xy_up: np.ndarray, side: str, d: int, leg_len: float, fps: float, cfg: EventsCfg
) -> np.ndarray:
    """Per-frame bool: foot near ground level AND moving backward with the belt.

    Parameters
    ----------
    xy_up : (T, J, 2) cleaned keypoints, y-up pixels.
    side : "l" or "r".
    d : facing direction (+1 / -1).
    leg_len : leg length in pixels (geometry.leg_length).
    fps : frames per second.
    cfg : EventsCfg.
    """
    sj = side_joints(side)
    heel, toe = xy_up[:, sj["heel"]], xy_up[:, sj["toe"]]
    foot_low = np.fmin(heel[:, 1], toe[:, 1])
    if not np.isfinite(foot_low).any():
        return np.zeros(len(foot_low), dtype=bool)
    ground = np.nanpercentile(foot_low, cfg.ground_percentile)
    near_ground = foot_low < ground + cfg.contact_height_frac * leg_len
    vx = np.gradient(heel[:, 0]) * fps
    backward = d * vx < 0
    mask = near_ground & backward
    mask &= np.isfinite(foot_low) & np.isfinite(vx)
    return mask


def detect_events(
    xy_up: np.ndarray, side: str, d: int, leg_len: float, fps: float, cfg: EventsCfg
) -> Events:
    """Detect IC/TO for one foot.

    Contact runs shorter than ``cfg.min_stance_s`` are dropped, as are runs touching the first
    or last frame (truncated stances). IC = first frame of a run, TO = last frame.
    """
    mask = contact_mask(xy_up, side, d, leg_len, fps, cfg)
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1  # inclusive
    min_len = max(1, round(cfg.min_stance_s * fps))
    n = len(mask)
    keep = ((ends - starts + 1) >= min_len) & (starts > 0) & (ends < n - 1)
    return Events(side=side, ic=starts[keep].astype(int), to=ends[keep].astype(int))


def stride_pairs(events: Events, t: np.ndarray, cfg: EventsCfg) -> list[tuple[int, int, int]]:
    """(ic_k, to_k, ic_{k+1}) for consecutive ICs whose stride time is within config bounds.

    Uses the timestamp array ``t`` (seconds), so variable-frame-rate video is handled.
    """
    out: list[tuple[int, int, int]] = []
    for k in range(len(events.ic) - 1):
        ic, nxt = int(events.ic[k]), int(events.ic[k + 1])
        st = float(t[nxt] - t[ic])
        if cfg.stride_time_min_s <= st <= cfg.stride_time_max_s:
            out.append((ic, int(events.to[k]), nxt))
    return out
