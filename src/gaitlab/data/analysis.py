"""Keypoints → stride and window rows, for one clip or a whole session (TECHNICAL_PLAN §4.3-4.4).

Shared by ``gaitlab process`` (which writes the rows to parquet) and ``eval.own_validation``
(which recomputes them in memory from each backend's npz, so two backends never overwrite each
other's processed files).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gaitlab.config import PipelineConfig
    from gaitlab.data.sessions import SessionRow
    from gaitlab.types import KeypointSeq


def clip_rows(
    seq: KeypointSeq,
    cfg: PipelineConfig,
    session_id: str,
    clip_id: str,
    condition: str,
    speed_kmh: float | None,
    standing_seq: KeypointSeq | None,
) -> tuple[list[dict], list[dict]]:
    """One clip → (strides rows, windows rows), each tagged with the sequence's synthetic flag."""
    from gaitlab.gait.features import analyze_clip, knee_cycle_curve
    from gaitlab.gait.windows import make_windows

    res = analyze_clip(seq, cfg, standing=standing_seq)
    flag = {"synthetic": seq.synthetic}
    strides = [
        {
            "session_id": session_id,
            "clip_id": clip_id,
            "condition": condition,
            **s.__dict__,
            "knee_flex_cycle_deg": knee_cycle_curve(
                res.xy_up, seq.t, res.side, s.ic_frame, s.next_ic_frame
            ),
            **flag,
        }
        for s in res.strides
    ]
    wins = make_windows(res.strides, session_id, clip_id, cfg.windows, condition, speed_kmh)
    return strides, [{**w.flat(), **flag} for w in wins]


@dataclass
class SessionRows:
    """Every running clip of one session, analyzed: rows plus what was read."""

    strides: list[dict] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)
    synthetic: set[bool] = field(default_factory=set)
    npz: list[Path] = field(default_factory=list)


def npz_path(interim: Path, clip_id: str, backend: str) -> Path:
    """``<interim>/<clip>.<backend>.npz``; FileNotFoundError naming the fix if it's missing."""
    p = interim / f"{clip_id}.{backend}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `gaitlab extract --backend {backend}` first")
    return p


def analyze_session(
    rows: Sequence[SessionRow], interim: Path, backend: str, cfg: PipelineConfig
) -> SessionRows:
    """Analyze every non-standing clip of one session from ``<interim>/<clip>.<backend>.npz``.

    The session's standing clip (if any) calibrates foot-strike angle, as in ``process``.
    """
    from gaitlab.types import KeypointSeq

    out = SessionRows()
    standing_row = next((r for r in rows if r.condition == "standing"), None)
    standing_seq = None
    if standing_row is not None:
        p = npz_path(interim, standing_row.clip_id, backend)
        standing_seq = KeypointSeq.load(str(p))
        out.npz.append(p)
    for r in rows:
        if r.condition == "standing":
            continue
        p = npz_path(interim, r.clip_id, backend)
        seq = KeypointSeq.load(str(p))
        out.npz.append(p)
        out.synthetic.add(seq.synthetic)
        s_rows, w_rows = clip_rows(
            seq, cfg, r.session_id, r.clip_id, r.condition, r.speed_kmh, standing_seq
        )
        out.strides.extend(s_rows)
        out.windows.extend(w_rows)
    return out
