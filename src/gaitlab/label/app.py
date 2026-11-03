"""Gradio labeling app (TECHNICAL_PLAN §6, US-5). TASKS M2.

Contract:
``launch(windows_glob="data/processed/*/windows.parquet", labels_csv="data/labels/labels.csv",
         blind: int | None = None, rater: str = "vw", seed: int = 0)``
- Normal mode: queue = windows without a normal-mode label, ordered by session then t_start.
  Show the pre-rendered overlay clip for [t_start, t_end] (``viz.overlay`` with time range),
  a feature table (median, IQR), grade radio 1–5, tag checkboxes (configs/features.yaml
  vocabulary), cue textbox, Save / Skip / Back.
- Blind mode (``blind=N``): sample N already-labeled windows with ``seed``; hide previous labels
  and the feature table; write rows with ``mode="blind_regrade"``.
- Writes: append-only to labels.csv with a file lock (``fcntl.flock`` on Linux); validate each
  row with ``data.sessions.LabelRow`` before writing. Never rewrite existing rows.
- Keyboard: 1–5 set the grade, Enter saves.
"""

from __future__ import annotations


def launch(blind: int | None = None, rater: str = "vw", seed: int = 0) -> None:
    raise NotImplementedError("TASKS.md M2: labeling app")
