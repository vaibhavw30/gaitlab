"""sessions.csv and labels.csv schemas and loaders (TECHNICAL_PLAN §4.5–4.6)."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Condition = Literal["natural", "overstride", "metronome", "lean", "fatigue", "standing"]


class SessionRow(BaseModel):
    session_id: str = Field(pattern=r"^\d{8}-\d{2}$")
    date: date
    clip_id: str
    path: str
    condition: Condition
    speed_kmh: float | None = None
    incline_pct: float | None = None
    shoes: str = ""
    fps: float
    camera_height_m: float | None = None
    camera_distance_m: float | None = None
    notes: str = ""


class LabelRow(BaseModel):
    window_id: str
    grade: int = Field(ge=1, le=5)
    tags: str = ""  # semicolon-separated, from configs/features.yaml vocabulary
    cue: str = ""
    rater: str
    mode: Literal["normal", "blind_regrade"] = "normal"
    labeled_at: datetime


def _drop_blank(row: dict[str, str]) -> dict[str, str]:
    """Blank CSV cells → field defaults (None for optional numbers, "" for text)."""
    return {k: v for k, v in row.items() if v not in ("", None)}


def load_sessions(path: str | Path = "data/sessions.csv") -> list[SessionRow]:
    """Rows of a sessions CSV. Extra columns (e.g. ``synthetic``) are ignored."""
    with open(path, newline="") as f:
        return [SessionRow.model_validate(_drop_blank(r)) for r in csv.DictReader(f)]


def load_labels(path: str | Path = "data/labels/labels.csv") -> list[LabelRow]:
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            r = {k: (v if v is not None else "") for k, v in r.items()}
            rows.append(LabelRow.model_validate(r))
        return rows


def latest_normal_labels(rows: list[LabelRow]) -> dict[str, LabelRow]:
    """Latest normal-mode label per window (append-only CSV; later rows win)."""
    out: dict[str, LabelRow] = {}
    for r in sorted(rows, key=lambda x: x.labeled_at):
        if r.mode == "normal":
            out[r.window_id] = r
    return out


LABEL_FIELDS: tuple[str, ...] = tuple(LabelRow.model_fields)
"""labels.csv columns, in order (TECHNICAL_PLAN §4.5)."""


def append_label(path: str | Path, row: LabelRow) -> None:
    """Append one validated row to labels.csv under an exclusive file lock.

    Append-only (TECHNICAL_PLAN §6): earlier rows are never rewritten; a correction is a new row
    and the later one wins (``latest_normal_labels``). The header is written once, by whichever
    writer finds the file empty while holding the lock. A file whose header isn't
    ``LABEL_FIELDS`` is refused untouched. ``fcntl.flock`` works on Linux and macOS.
    """
    import fcntl
    import os

    row = LabelRow.model_validate(row.model_dump())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            empty = f.tell() == 0
            if not empty:
                f.seek(0)
                header = f.readline().rstrip("\r\n")
                if header != ",".join(LABEL_FIELDS):
                    raise ValueError(
                        f"{path}: header {header!r} != {','.join(LABEL_FIELDS)!r}; not appending"
                    )
                f.seek(0, os.SEEK_END)  # back to the end before switching from reading to writing
            w = csv.DictWriter(f, fieldnames=list(LABEL_FIELDS))
            if empty:
                w.writeheader()
            w.writerow(row.model_dump(mode="json"))
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
