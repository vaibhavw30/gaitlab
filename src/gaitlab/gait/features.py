"""Per-stride features (TECHNICAL_PLAN §5.3). Pure functions over y-up pixel arrays.

Sign conventions (d = facing direction):
- overstride_L > 0: heel lands ahead of the hip.
- shin_angle_deg > 0: ankle ahead of knee at contact (shin reaching forward).
- foot_strike_deg > 0: toe up (dorsiflexed) relative to the standing foot angle.
- trunk_lean_deg > 0: shoulders ahead of hips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaitlab.config import PipelineConfig
from gaitlab.gait import geometry
from gaitlab.gait.events import detect_events, stride_pairs
from gaitlab.pose.skeleton import side_joints
from gaitlab.signal.clean import clean_keypoints
from gaitlab.types import Events, KeypointSeq, Stride

NEEDED = ("shoulder", "hip", "knee", "ankle", "heel", "toe")


def foot_angle_deg(xy_up: np.ndarray, side: str, d: int) -> np.ndarray:
    """Per-frame heel→toe angle vs horizontal (deg), positive when the toe is up."""
    sj = side_joints(side)
    heel, toe = xy_up[:, sj["heel"]], xy_up[:, sj["toe"]]
    return np.degrees(np.arctan2(toe[:, 1] - heel[:, 1], d * (toe[:, 0] - heel[:, 0])))


def foot_strike_offset_from_standing(xy_up_standing: np.ndarray, side: str, d: int) -> float:
    """Median standing foot angle (deg) to subtract from strike angles (Altman & Davis 2012)."""
    return float(np.nanmedian(foot_angle_deg(xy_up_standing, side, d)))


def knee_flexion_deg(xy_up: np.ndarray, side: str) -> np.ndarray:
    """Per-frame knee flexion = 180 − hip-knee-ankle angle."""
    sj = side_joints(side)
    return 180.0 - geometry.joint_angle_deg(
        xy_up[:, sj["hip"]], xy_up[:, sj["knee"]], xy_up[:, sj["ankle"]]
    )


def shin_angle_deg(xy_up: np.ndarray, side: str, d: int) -> np.ndarray:
    """Per-frame shin angle from vertical (deg), positive when the ankle is ahead of the knee."""
    sj = side_joints(side)
    knee, ankle = xy_up[:, sj["knee"]], xy_up[:, sj["ankle"]]
    return np.degrees(np.arctan2(d * (ankle[:, 0] - knee[:, 0]), knee[:, 1] - ankle[:, 1]))


def knee_cycle_curve(
    xy_up: np.ndarray, t: np.ndarray, side: str, ic: int, next_ic: int, n: int = 101
) -> np.ndarray:
    """Knee flexion (deg) over one stride, resampled by time to ``n`` points (0–100%).

    Point 0 is the IC frame and point ``n − 1`` the next IC frame; ``t`` (s) handles VFR.
    """
    seg = slice(int(ic), int(next_ic) + 1)
    knee = knee_flexion_deg(xy_up[seg], side)
    tt = np.asarray(t[seg], dtype=np.float64)
    frac = (tt - tt[0]) / (tt[-1] - tt[0])
    return np.interp(np.linspace(0.0, 1.0, n), frac, knee)


def stride_features(
    xy_up: np.ndarray,
    t: np.ndarray,
    side: str,
    d: int,
    leg_len: float,
    events: Events,
    cfg: PipelineConfig,
    standing_offset_deg: float | None = None,
) -> list[Stride]:
    """Compute one ``Stride`` per valid (ic, to, next_ic) triple. Pixels in, L/deg/s out."""
    sj = side_joints(side)
    hip, heel, shoulder = xy_up[:, sj["hip"]], xy_up[:, sj["heel"]], xy_up[:, sj["shoulder"]]
    kflex = knee_flexion_deg(xy_up, side)
    shin = shin_angle_deg(xy_up, side, d)
    fang = foot_angle_deg(xy_up, side, d)
    trunk = geometry.signed_angle_from_vertical_deg(hip, shoulder, d)
    needed_idx = [sj[p] for p in NEEDED]
    finite = np.isfinite(xy_up[:, needed_idx]).all(axis=(1, 2))

    strides: list[Stride] = []
    for k, (ic, to, nxt) in enumerate(stride_pairs(events, t, cfg.events)):
        stride_time = float(t[nxt] - t[ic])
        frame_dt = stride_time / (nxt - ic)
        contact_time = (to - ic + 1) * frame_dt
        strike = float(fang[ic])
        calibrated = standing_offset_deg is not None
        if calibrated:
            strike -= float(standing_offset_deg)
        strides.append(
            Stride(
                stride_idx=k,
                side=side,
                ic_frame=ic,
                to_frame=to,
                next_ic_frame=nxt,
                t_ic=float(t[ic]),
                stride_time_s=stride_time,
                contact_time_s=contact_time,
                duty_factor=contact_time / stride_time,
                cadence_spm=120.0 / stride_time,
                overstride_L=float(d * (heel[ic, 0] - hip[ic, 0]) / leg_len),
                shin_angle_deg=float(shin[ic]),
                knee_flex_ic_deg=float(kflex[ic]),
                knee_flex_peak_deg=float(np.nanmax(kflex[ic : to + 1])),
                foot_strike_deg=strike,
                foot_strike_calibrated=calibrated,
                vert_osc_L=float((np.nanmax(hip[ic:nxt, 1]) - np.nanmin(hip[ic:nxt, 1])) / leg_len),
                trunk_lean_deg=float(np.nanmean(trunk[ic : to + 1])),
                quality=float(finite[ic:nxt].mean()),
            )
        )
    return strides


@dataclass
class ClipAnalysis:
    """One clip's analysis. ``xy_up``: feature-filtered keypoints (T, J, 2), y-up px."""

    side: str
    d: int
    leg_len_px: float
    events: Events
    strides: list[Stride]
    xy_up: np.ndarray | None = None


def analyze_clip(
    seq: KeypointSeq,
    cfg: PipelineConfig,
    standing: KeypointSeq | None = None,
) -> ClipAnalysis:
    """Clean → near side → direction → leg length → events → per-stride features.

    Events use the ``lowpass_hz``-filtered signal; features use the ``feature_lowpass_hz``
    signal (ADR-0008).
    """
    h = seq.size[1]
    xy_ev = geometry.to_y_up(clean_keypoints(seq.xy, seq.vis, seq.fps, cfg.clean), h)
    xy_ft = geometry.to_y_up(
        clean_keypoints(seq.xy, seq.vis, seq.fps, cfg.clean, for_features=True), h
    )
    side = geometry.near_side(seq.vis)
    d = geometry.facing_direction(xy_ev, side)
    leg_len = geometry.leg_length(xy_ev, side)
    events = detect_events(xy_ev, side, d, leg_len, seq.fps, cfg.events)
    offset = None
    if standing is not None:
        st = clean_keypoints(standing.xy, standing.vis, standing.fps, cfg.clean, for_features=True)
        offset = foot_strike_offset_from_standing(geometry.to_y_up(st, standing.size[1]), side, d)
    strides = stride_features(xy_ft, seq.t, side, d, leg_len, events, cfg, offset)
    return ClipAnalysis(
        side=side, d=d, leg_len_px=leg_len, events=events, strides=strides, xy_up=xy_ft
    )
