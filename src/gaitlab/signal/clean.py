"""Keypoint cleaning: visibility gating, short-gap interpolation, zero-phase low-pass.

Pure functions (TECHNICAL_PLAN §5.1). Input and output are (T, J, 2) arrays; NaN = missing.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from gaitlab.config import CleanCfg


def gate(xy: np.ndarray, vis: np.ndarray, visibility_min: float) -> np.ndarray:
    """Set joints with ``vis < visibility_min`` to NaN. Returns a float64 copy."""
    out = np.array(xy, dtype=np.float64, copy=True)
    out[vis < visibility_min] = np.nan
    return out


def _nan_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) for each run of True in a 1-D bool array."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def interpolate_short_gaps(series: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill interior NaN runs of length ≤ ``max_gap`` in a 1-D series."""
    out = np.array(series, dtype=np.float64, copy=True)
    n = len(out)
    for s, e in _nan_runs(np.isnan(out)):
        if s == 0 or e == n or (e - s) > max_gap:
            continue
        out[s:e] = np.interp(np.arange(s, e), [s - 1, e], [out[s - 1], out[e]])
    return out


def lowpass(series: np.ndarray, fps: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass applied per contiguous finite segment.

    Segments too short for ``filtfilt``'s padding are returned unfiltered.
    """
    out = np.array(series, dtype=np.float64, copy=True)
    if cutoff_hz >= fps / 2:
        return out
    b, a = butter(order, cutoff_hz / (fps / 2), btype="low")
    padlen = 3 * max(len(a), len(b))
    for s, e in _nan_runs(np.isfinite(out)):
        if e - s > padlen:
            out[s:e] = filtfilt(b, a, out[s:e])
    return out


def clean_keypoints(
    xy: np.ndarray,
    vis: np.ndarray,
    fps: float,
    cfg: CleanCfg,
    *,
    for_features: bool = False,
) -> np.ndarray:
    """Gate → interpolate short gaps → low-pass, for every joint and axis.

    Parameters
    ----------
    xy : (T, J, 2) pixels (either y convention; the operation is axis-independent).
    vis : (T, J) visibility.
    fps : frames per second.
    cfg : CleanCfg.
    for_features : use ``cfg.feature_lowpass_hz`` instead of ``cfg.lowpass_hz``. Heavy
        low-pass rounds the sharp reversal of the foot at contact and biases contact-instant
        features (ADR-0008), so events and features use different cutoffs.

    Returns
    -------
    (T, J, 2) float64 with NaN where data is missing for longer than ``cfg.max_gap_s``.
    """
    cutoff = cfg.feature_lowpass_hz if for_features else cfg.lowpass_hz
    max_gap = max(1, round(cfg.max_gap_s * fps))
    out = gate(xy, vis, cfg.visibility_min)
    _, n_j, _ = out.shape
    for j in range(n_j):
        for k in range(2):
            s = interpolate_short_gaps(out[:, j, k], max_gap)
            if cutoff is not None:
                s = lowpass(s, fps, cutoff, cfg.filter_order)
            out[:, j, k] = s
    return out
