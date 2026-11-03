"""Deterministic prompt and target formatting (TECHNICAL_PLAN §7.2). Pure functions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

INSTRUCTION = "Grade this treadmill running window from 1 (poor) to 5 (excellent)."
GRADE_PREFIX = "grade:"

# Display precision per feature (fixed so the prompt is byte-stable).
PRECISION: dict[str, int] = {
    "cadence_spm": 1,
    "overstride_L": 3,
    "shin_angle_deg": 1,
    "knee_flex_ic_deg": 1,
    "knee_flex_peak_deg": 1,
    "foot_strike_deg": 1,
    "vert_osc_L": 3,
    "trunk_lean_deg": 1,
    "contact_time_s": 3,
    "duty_factor": 3,
}


def _num(v: float | None, prec: int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    return f"{v:.{prec}f}"


def format_prompt(
    window_row: Mapping[str, float | str | int | None],
    feature_order: Sequence[str],
) -> str:
    """Render a window (windows.parquet row, i.e. ``Window.flat()``) as the user prompt.

    One line per feature in ``feature_order``: ``<name>: <median> (iqr <iqr>)``.
    Speed is included when known.
    """
    lines = [INSTRUCTION]
    speed = window_row.get("speed_kmh")
    if speed is not None and not (isinstance(speed, float) and math.isnan(speed)):
        lines.append(f"speed_kmh: {float(speed):.1f}")
    for f in feature_order:
        prec = PRECISION.get(f, 3)
        med = window_row.get(f"{f}_med")
        iqr = window_row.get(f"{f}_iqr")
        lines.append(f"{f}: {_num(med, prec)} (iqr {_num(iqr, prec)})")  # type: ignore[arg-type]
    return "\n".join(lines)


def format_target(grade: int, tags: Sequence[str], cue: str) -> str:
    """Assistant target. ``grade:`` comes first so it can be prefilled for logit scoring."""
    if grade not in (1, 2, 3, 4, 5):
        raise ValueError(f"grade must be 1..5, got {grade}")
    tag_str = "; ".join(tags) if tags else "none"
    return f"{GRADE_PREFIX} {grade}\ntags: {tag_str}\ncue: {cue.strip()}"


@dataclass
class ParsedOutput:
    grade: int | None
    tags: list[str]
    cue: str


_GRADE_RE = re.compile(r"grade:\s*([1-5])")
_TAGS_RE = re.compile(r"tags:\s*(.*)")
_CUE_RE = re.compile(r"cue:\s*(.*)", re.S)


def parse_output(text: str) -> ParsedOutput:
    """Parse generated text. Used for tags and cue; the grade itself comes from logits."""
    g = _GRADE_RE.search(text)
    t = _TAGS_RE.search(text)
    c = _CUE_RE.search(text)
    tags: list[str] = []
    if t:
        raw = t.group(1).strip()
        if raw and raw.lower() != "none":
            tags = [s.strip() for s in raw.split(";") if s.strip()]
    return ParsedOutput(
        grade=int(g.group(1)) if g else None,
        tags=tags,
        cue=(c.group(1).strip().splitlines() or [""])[0] if c else "",
    )
