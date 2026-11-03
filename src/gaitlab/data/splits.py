"""Session-level splits (ADR-0003). Never split windows randomly."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path


def make_session_split(
    session_ids_with_dates: Sequence[tuple[str, str]], n_test: int = 2, n_dev: int = 1
) -> dict[str, str]:
    """Assign whole sessions by date: last ``n_test`` → test, next ``n_dev`` → dev, rest → train.

    Parameters
    ----------
    session_ids_with_dates : (session_id, ISO date) pairs; duplicates are collapsed.
    """
    uniq = sorted(set(session_ids_with_dates), key=lambda x: (x[1], x[0]))
    if len(uniq) < n_test + n_dev + 1:
        raise ValueError(f"need at least {n_test + n_dev + 1} sessions, got {len(uniq)}")
    out: dict[str, str] = {}
    for i, (sid, _) in enumerate(uniq):
        from_end = len(uniq) - i
        if from_end <= n_test:
            out[sid] = "test"
        elif from_end <= n_test + n_dev:
            out[sid] = "dev"
        else:
            out[sid] = "train"
    return out


def write_split_frozen(split: dict[str, str], path: str | Path = "data/splits.json") -> Path:
    """Write once. Refuses to overwrite an existing, different split."""
    p = Path(path)
    if p.exists():
        existing = json.loads(p.read_text())
        if existing != split:
            raise FileExistsError(f"{p} is frozen and differs from the new split; not overwriting")
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    return p


def leave_one_session_out(session_ids: Sequence[str]) -> list[tuple[list[int], list[int]]]:
    """(train_idx, val_idx) folds holding out one session at a time."""
    ids = list(session_ids)
    folds = []
    for held in sorted(set(ids)):
        val = [i for i, s in enumerate(ids) if s == held]
        train = [i for i, s in enumerate(ids) if s != held]
        folds.append((train, val))
    return folds
