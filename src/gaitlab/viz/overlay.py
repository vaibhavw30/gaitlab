"""Skeleton overlay video / GIF (US-1). TASKS M0.

Contract:
``render_overlay(video_path, seq: KeypointSeq, analysis: ClipAnalysis, out_path, gif=False,
                 max_seconds=None) -> Path``
- Draw with Pillow ``ImageDraw`` on decoded frames (``io.video.iter_frames``): canonical bones
  (shoulder–hip–knee–ankle–heel–toe per side, near side brighter), joints coloured by visibility.
- Mark IC frames (a ring at the heel, held for ~0.1 s) and TO frames (a different marker).
- Text box: knee flexion and shin angle for the near side, live; cadence of the current stride.
- Encode mp4 with PyAV (H.264, ``preset=veryfast``) or a GIF (downscale to 480 px wide, 15 fps,
  via Pillow ``save(..., save_all=True)``).
Frames are 1:1 with ``seq.t``; assert lengths match.
"""

from __future__ import annotations

from pathlib import Path


def render_overlay(
    video_path: str,
    seq,
    analysis,
    out_path: str,
    gif: bool = False,
    max_seconds: float | None = None,
) -> Path:
    raise NotImplementedError("TASKS.md M0: overlay")
