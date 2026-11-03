"""Build SFT JSONL from labeled windows (TECHNICAL_PLAN §4.7). Pure except ``write_jsonl``."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gaitlab.llm.prompt import format_prompt, format_target


def build_records(
    windows: Iterable[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, str],
    feature_order: Sequence[str],
    synthetic: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Join windows with labels (normal-mode only) and route by the session split.

    Parameters
    ----------
    windows : rows of windows.parquet (dict-like, includes ``window_id`` and ``session_id``).
    labels : window_id → {"grade": int, "tags": "a;b", "cue": str}; blind re-grade rows excluded.
    split : session_id → train | dev | test.
    """
    out: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for w in windows:
        lab = labels.get(str(w["window_id"]))
        if lab is None:
            continue
        part = split.get(str(w["session_id"]))
        if part is None:
            raise KeyError(f"session {w['session_id']} missing from split")
        tags = [t for t in str(lab.get("tags") or "").split(";") if t]
        out[part].append(
            {
                "window_id": w["window_id"],
                "messages": [
                    {"role": "user", "content": format_prompt(w, feature_order)},
                    {
                        "role": "assistant",
                        "content": format_target(int(lab["grade"]), tags, str(lab.get("cue", ""))),
                    },
                ],
                "synthetic": synthetic,
            }
        )
    return out


def write_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p
