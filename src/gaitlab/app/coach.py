"""Gradio coach demo (US-1..US-4). TASKS M4.

Contract: upload a clip (or pick a session) → extract (configured backend) → process → windows →
grade (logits) + tags + cue → verifier. Show the overlay video, a per-window table (grade,
P(grade), tags), and the cue only when the verifier passes (otherwise show
"no grounded cue" and log the reasons). A "compare to session" dropdown shows feature deltas
with the session's IQR. All displayed metrics are computed live from the uploaded clip; no
stored benchmark numbers are shown in the UI.
"""

from __future__ import annotations


def launch() -> None:
    raise NotImplementedError("TASKS.md M4: coach app")
