"""Group strides into fixed-size windows with median/IQR summaries (TECHNICAL_PLAN §5.4)."""

from __future__ import annotations

import numpy as np

from gaitlab.config import WindowsCfg
from gaitlab.types import FEATURE_FIELDS, Stride, Window


def _iqr(x: np.ndarray) -> float:
    q75, q25 = np.nanpercentile(x, [75, 25])
    return float(q75 - q25)


def make_windows(
    strides: list[Stride],
    session_id: str,
    clip_id: str,
    cfg: WindowsCfg,
    condition: str = "",
    speed_kmh: float | None = None,
) -> list[Window]:
    """Non-overlapping windows of ``cfg.strides`` consecutive strides from one clip.

    "Consecutive" means stride k's next IC is stride k+1's IC; a gap (a dropped stride) starts a
    new run. Windows whose mean stride quality is below ``cfg.quality_min`` are dropped. A
    trailing partial window is dropped.
    """
    runs: list[list[Stride]] = []
    for s in strides:
        if runs and runs[-1][-1].next_ic_frame == s.ic_frame:
            runs[-1].append(s)
        else:
            runs.append([s])

    windows: list[Window] = []
    n = cfg.strides
    for run in runs:
        for start in range(0, len(run) - n + 1, n):
            chunk = run[start : start + n]
            quality = float(np.mean([s.quality for s in chunk]))
            if quality < cfg.quality_min:
                continue
            med, iqr = {}, {}
            for f in FEATURE_FIELDS:
                vals = np.array([getattr(s, f) for s in chunk], dtype=np.float64)
                med[f] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
                iqr[f] = _iqr(vals) if np.isfinite(vals).any() else float("nan")
            first = chunk[0]
            windows.append(
                Window(
                    window_id=f"{session_id}:{clip_id}:{first.stride_idx}",
                    session_id=session_id,
                    clip_id=clip_id,
                    start_stride=first.stride_idx,
                    n_strides=n,
                    t_start=first.t_ic,
                    t_end=chunk[-1].t_ic + chunk[-1].stride_time_s,
                    quality=quality,
                    med=med,
                    iqr=iqr,
                    condition=condition,
                    speed_kmh=speed_kmh,
                )
            )
    return windows
