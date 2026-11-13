"""Labeling logic behind the app (TECHNICAL_PLAN §6, US-5, FR-7). TASKS M2.

Pure: no Gradio, no file I/O. Rows are written through the injected ``write`` callable
(``data.sessions.append_label`` in the app).

- ``normal_queue``: windows without a normal-mode label, by session, then clip (in the order the
  windows table lists them), then ``t_start``.
- ``blind_sample``: ``n`` normal-labeled windows drawn with ``seed`` (sorted ids, so the draw
  depends only on which windows are labeled), minus those already blind re-graded, so a restart
  with the same seed resumes the same sample.
- ``LabelSession``: the current window, Save / Skip / Back, progress. In blind mode the title
  hides the window id (a clip id such as ``overstride`` names the condition) and there is no
  feature table.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from gaitlab.data.sessions import LabelRow

MODES = ("normal", "blind_regrade")


def normal_queue(windows: Sequence[Mapping[str, Any]], labels: Iterable[LabelRow]) -> list[str]:
    """Window ids still needing a normal-mode label, by session, then clip, then time."""
    labeled = {r.window_id for r in labels if r.mode == "normal"}
    clip_order: dict[tuple[str, str], int] = {}
    for w in windows:
        clip_order.setdefault((str(w["session_id"]), str(w["clip_id"])), len(clip_order))

    def key(w: Mapping[str, Any]) -> tuple:
        sid, cid = str(w["session_id"]), str(w["clip_id"])
        return sid, clip_order[(sid, cid)], float(w["t_start"])

    return [str(w["window_id"]) for w in sorted(windows, key=key) if w["window_id"] not in labeled]


def blind_sample(labels: Sequence[LabelRow], n: int, seed: int) -> list[str]:
    """Seeded sample of ``n`` normal-labeled windows, minus those already blind re-graded."""
    if n <= 0:
        raise ValueError(f"blind re-grade needs n > 0, got {n}")
    labeled = sorted({r.window_id for r in labels if r.mode == "normal"})
    if not labeled:
        return []
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(labeled), size=min(n, len(labeled)), replace=False)
    done = {r.window_id for r in labels if r.mode == "blind_regrade"}
    return [labeled[i] for i in picked if labeled[i] not in done]


def check_synthetic(flags: set[bool], synthetic_mode: bool) -> None:
    """Synthetic windows are labeled only into the synthetic tree (CLAUDE.md rule 3)."""
    if not flags:
        raise ValueError("no windows to label: run `gaitlab process` first")
    if len(flags) > 1:
        raise ValueError("these windows mix synthetic and real sessions; label them separately")
    if flags == {True} and not synthetic_mode:
        raise ValueError(
            "these windows are synthetic: use --synthetic (labels go under data/synthetic/)"
        )
    if flags == {False} and synthetic_mode:
        raise ValueError("these windows are not synthetic; drop --synthetic")


def make_label(
    window_id: str,
    grade: int | str | None,
    tags: Iterable[str],
    cue: str,
    rater: str,
    mode: str,
    vocab: Sequence[str],
    now: datetime,
) -> LabelRow:
    """A validated label row. Tags are stored in vocabulary order; the cue is one line."""
    if grade is None or str(grade).strip() == "":
        raise ValueError("pick a grade (1-5) before saving")
    g = int(grade)
    if not 1 <= g <= 5:
        raise ValueError(f"grade must be 1-5, got {g}")
    chosen = set(tags)
    unknown = sorted(chosen - set(vocab))
    if unknown:
        raise ValueError(f"unknown tag(s) {unknown}; vocabulary is configs/features.yaml")
    if not rater.strip():
        raise ValueError("rater must not be empty")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return LabelRow(
        window_id=window_id,
        grade=g,
        tags=";".join(t for t in vocab if t in chosen),
        cue=" ".join(cue.split()),
        rater=rater.strip(),
        mode=mode,  # type: ignore[arg-type]
        labeled_at=now,
    )


def feature_rows(window: Mapping[str, Any], feature_order: Sequence[str]) -> list[list]:
    """[[feature, median, IQR], ...] in ``feature_order``; missing or NaN values are None."""

    def val(key: str) -> float | None:
        v = window.get(key)
        return None if v is None or not np.isfinite(float(v)) else float(v)

    return [[f, val(f"{f}_med"), val(f"{f}_iqr")] for f in feature_order]


class LabelSession:
    """Where the labeler is in the queue, and what Save / Skip / Back do.

    Parameters
    ----------
    window_ids : the queue (``normal_queue`` or ``blind_sample``).
    windows : window id → windows-table row.
    mode : "normal" | "blind_regrade".
    write : called with each saved ``LabelRow`` (append to labels.csv).
    """

    def __init__(
        self,
        window_ids: Sequence[str],
        windows: Mapping[str, Mapping[str, Any]],
        *,
        mode: str,
        rater: str,
        vocab: Sequence[str],
        write: Callable[[LabelRow], None],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.ids = list(window_ids)
        self.windows = dict(windows)
        self.mode = mode
        self.rater = rater
        self.vocab = list(vocab)
        self._write = write
        self._now = now
        self.i = 0
        self.saved = 0

    @property
    def blind(self) -> bool:
        return self.mode == "blind_regrade"

    @property
    def done(self) -> bool:
        return self.i >= len(self.ids)

    def current(self) -> str | None:
        return None if self.done else self.ids[self.i]

    def window(self) -> Mapping[str, Any] | None:
        wid = self.current()
        return None if wid is None else self.windows[wid]

    def title(self) -> str:
        if self.done:
            return f"Done: {self.saved} saved"
        pos = f"{self.i + 1} / {len(self.ids)}"
        return f"Blind re-grade {pos}" if self.blind else f"{self.current()} ({pos})"

    def progress(self) -> str:
        pos = f"{min(self.i + 1, len(self.ids))} / {len(self.ids)}"
        return f"{pos} · {self.saved} saved"

    def features(self, feature_order: Sequence[str]) -> list[list]:
        w = self.window()
        return [] if self.blind or w is None else feature_rows(w, feature_order)

    def save(self, grade: int | str | None, tags: Iterable[str], cue: str) -> LabelRow:
        wid = self.current()
        if wid is None:
            raise ValueError("no window left to save")
        row = make_label(wid, grade, tags, cue, self.rater, self.mode, self.vocab, self._now())
        self._write(row)
        self.saved += 1
        self.i += 1
        return row

    def skip(self) -> None:
        self.i = min(self.i + 1, len(self.ids))

    def back(self) -> None:
        self.i = max(self.i - 1, 0)
