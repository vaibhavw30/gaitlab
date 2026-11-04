"""Skeleton overlay video / GIF (US-1). TASKS M0.

``render_overlay(seq, analysis, out_path, video_path=None, gif=False, max_seconds=None) -> Path``

- Background: the decoded video (``io.video.iter_frames``) if ``video_path`` is given, otherwise
  a plain canvas of ``seq.size``. Video frames are 1:1 with ``seq.t``; a frame-count or size
  mismatch raises ``ValueError``.
- Skeleton: the keypoints as the backend stored them (``seq.xy``, y-down px), shoulder–hip–
  knee–ankle–heel–toe per side plus the nose, near side brighter; joints coloured by
  visibility (red at 0 → green at 1). Missing joints are skipped.
- Events (``analysis.events``, near side): a ring at the heel from each IC frame and a square at
  the toe from each TO frame, held for ``hold_s`` (0.1 s).
- Text: near-side knee flexion and shin angle for the frame, from the feature-filtered signal the
  stride features use (``analysis.xy_up``), and the cadence of the stride containing the frame.
- Synthetic sequences (``seq.synthetic``) get a "SYNTHETIC" watermark on every frame.
- Output: mp4 via PyAV (H.264, ``preset=veryfast``, PTS from ``seq.t`` so VFR timing survives),
  or a GIF (downscaled to at most 480 px wide, sampled at 15 fps by timestamp, Pillow).

``annotate`` (what to draw per frame) is pure; Pillow and PyAV are imported lazily.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from gaitlab.gait.features import ClipAnalysis, knee_flexion_deg, shin_angle_deg
from gaitlab.pose.skeleton import J
from gaitlab.types import KeypointSeq

HOLD_S = 0.1
GIF_FPS = 15.0
GIF_MAX_WIDTH = 480
MP4_TIME_BASE = Fraction(1, 90000)

CANVAS_COLOR = (24, 24, 28)
IC_COLOR = (255, 0, 255)
TO_COLOR = (0, 255, 255)
NEAR_BONE = (255, 210, 60)
FAR_BONE = (120, 105, 60)
TEXT_COLOR = (245, 245, 245)
WATERMARK_COLOR = (255, 90, 90)

_BONES = [
    ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
    ("ankle", "heel"), ("heel", "toe"), ("ankle", "toe"),
]  # fmt: skip


@dataclass(frozen=True)
class FrameAnnotation:
    """What the overlay shows on one frame.

    Attributes
    ----------
    frame : int
        Index into the clip.
    t : float
        Frame time, s.
    ic_age, to_age : int or None
        Frames since the latest near-side IC / TO if it is within the hold time, else None.
    knee_flex_deg, shin_angle_deg : float
        Near-side angles on this frame, deg (NaN where joints are missing).
    cadence_spm : float or None
        Cadence of the stride ``[ic, next_ic)`` containing the frame, steps/min.
    """

    frame: int
    t: float
    ic_age: int | None
    to_age: int | None
    knee_flex_deg: float
    shin_angle_deg: float
    cadence_spm: float | None


def _event_ages(frames: np.ndarray, t: np.ndarray, hold_s: float) -> list[int | None]:
    """Per frame: frames since the latest event at or before it, if within ``hold_s``."""
    ages: list[int | None] = [None] * len(t)
    ev = np.sort(np.asarray(frames, dtype=int))
    for k, e in enumerate(ev):
        end = int(ev[k + 1]) if k + 1 < len(ev) else len(t)
        for i in range(int(e), end):
            if i > e and t[i] - t[e] >= hold_s:
                break
            ages[i] = i - int(e)
    return ages


def annotate(
    seq: KeypointSeq, analysis: ClipAnalysis, hold_s: float = HOLD_S
) -> list[FrameAnnotation]:
    """Per-frame markers and near-side angles. Pure.

    Angles come from ``analysis.xy_up`` (the feature-filtered signal), so the value shown on an
    IC frame equals that stride's ``shin_angle_deg`` / ``knee_flex_ic_deg``.
    """
    if analysis.xy_up is None:
        raise ValueError("analysis has no xy_up; build it with gait.features.analyze_clip")
    if len(analysis.xy_up) != seq.n_frames:
        raise ValueError(f"analysis has {len(analysis.xy_up)} frames, seq has {seq.n_frames}")
    t = np.asarray(seq.t, dtype=np.float64)
    knee = knee_flexion_deg(analysis.xy_up, analysis.side)
    shin = shin_angle_deg(analysis.xy_up, analysis.side, analysis.d)
    ic_age = _event_ages(analysis.events.ic, t, hold_s)
    to_age = _event_ages(analysis.events.to, t, hold_s)
    cadence: list[float | None] = [None] * seq.n_frames
    for s in analysis.strides:
        for i in range(s.ic_frame, s.next_ic_frame):
            cadence[i] = s.cadence_spm
    return [
        FrameAnnotation(
            frame=i,
            t=float(t[i]),
            ic_age=ic_age[i],
            to_age=to_age[i],
            knee_flex_deg=float(knee[i]),
            shin_angle_deg=float(shin[i]),
            cadence_spm=cadence[i],
        )
        for i in range(seq.n_frames)
    ]


def gif_frame_indices(t: np.ndarray, gif_fps: float = GIF_FPS) -> np.ndarray:
    """First frame of each ``1 / gif_fps`` interval of ``t`` (s), so GIF timing follows PTS."""
    bucket = np.floor((np.asarray(t, dtype=np.float64) - t[0]) * gif_fps + 1e-9).astype(int)
    return np.flatnonzero(np.r_[True, np.diff(bucket) > 0])


def _vis_color(v: float) -> tuple[int, int, int]:
    v = float(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else 0.0
    return (round(230 - 170 * v), round(60 + 160 * v), 60)


def _fmt(x: float | None, unit: str) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:5.1f}{unit}"


def _draw_frame(img, seq: KeypointSeq, i: int, a: FrameAnnotation, side: str, font) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    w, h = seq.size
    lw = max(2, round(h / 270))
    r = max(3, round(h / 180))
    xy, vis = seq.xy[i], seq.vis[i]

    def pt(name: str) -> tuple[float, float] | None:
        p = xy[J[name]]
        return (float(p[0]), float(p[1])) if np.isfinite(p).all() else None

    far = "l" if side == "r" else "r"
    for s, color in ((far, FAR_BONE), (side, NEAR_BONE)):
        for a_, b_ in _BONES:
            pa, pb = pt(f"{s}_{a_}"), pt(f"{s}_{b_}")
            if pa and pb:
                draw.line([pa, pb], fill=color, width=lw)
    for s in (far, side):
        sh = pt(f"{s}_shoulder")
        nose = pt("nose")
        if sh and nose:
            draw.line([sh, nose], fill=NEAR_BONE if s == side else FAR_BONE, width=lw)
    for name, j in J.items():
        p = pt(name)
        if p:
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=_vis_color(vis[j]))

    if a.ic_age is not None and (heel := pt(f"{side}_heel")):
        rr = 4 * r
        draw.ellipse([heel[0] - rr, heel[1] - rr, heel[0] + rr, heel[1] + rr], outline=IC_COLOR,
                     width=lw)  # fmt: skip
    if a.to_age is not None and (toe := pt(f"{side}_toe")):
        rr = 3 * r
        draw.rectangle([toe[0] - rr, toe[1] - rr, toe[0] + rr, toe[1] + rr], outline=TO_COLOR,
                       width=lw)  # fmt: skip

    s = side.upper()
    lines = [
        f"{s} knee {_fmt(a.knee_flex_deg, '°')}   {s} shin {_fmt(a.shin_angle_deg, '°')}",
        f"cadence {_fmt(a.cadence_spm, ' spm')}   t {a.t:6.2f} s",
    ]
    pad = max(4, round(h / 100))
    box = draw.multiline_textbbox((pad, pad), "\n".join(lines), font=font)
    draw.rectangle([box[0] - pad // 2, box[1] - pad // 2, box[2] + pad // 2, box[3] + pad // 2],
                   fill=(0, 0, 0))  # fmt: skip
    draw.multiline_text((pad, pad), "\n".join(lines), fill=TEXT_COLOR, font=font)
    if seq.synthetic:
        mark = "SYNTHETIC"
        mb = draw.textbbox((0, 0), mark, font=font)
        draw.text((w - (mb[2] - mb[0]) - pad, pad), mark, fill=WATERMARK_COLOR, font=font)


def render_frames(
    seq: KeypointSeq,
    analysis: ClipAnalysis,
    background: Iterable[np.ndarray] | None = None,
    hold_s: float = HOLD_S,
    max_frames: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield one annotated RGB uint8 (H, W, 3) frame per keypoint frame.

    Parameters
    ----------
    seq, analysis
        The clip and its ``analyze_clip`` result.
    background : iterable of (H, W, 3) uint8, optional
        Video frames 1:1 with ``seq``; a plain canvas of ``seq.size`` if None.
    max_frames : int, optional
        Stop after this many frames (the background need only be that long).
    """
    from PIL import Image, ImageFont

    w, h = seq.size
    n = seq.n_frames if max_frames is None else min(max_frames, seq.n_frames)
    ann = annotate(seq, analysis, hold_s)
    font = ImageFont.load_default(size=max(10, round(h / 36)))
    bg_iter = iter(background) if background is not None else None
    for i in range(n):
        if bg_iter is None:
            img = Image.new("RGB", (w, h), CANVAS_COLOR)
        else:
            frame = next(bg_iter, None)
            if frame is None:
                raise ValueError(f"video has {i} frames, keypoints have {seq.n_frames}")
            if frame.shape[:2] != (h, w):
                raise ValueError(
                    f"video frame size {frame.shape[1]}x{frame.shape[0]} != keypoint size {w}x{h}"
                )
            img = Image.fromarray(np.asarray(frame, dtype=np.uint8), "RGB")
        _draw_frame(img, seq, i, ann[i], analysis.side, font)
        yield np.asarray(img)
    if bg_iter is not None and max_frames is None and next(bg_iter, None) is not None:
        raise ValueError(f"video has more frames than the keypoints ({seq.n_frames})")


def _write_mp4(frames: Iterator[np.ndarray], t: np.ndarray, fps: float, out: Path, meta: dict):
    import av

    with av.open(str(out), "w") as c:
        for k, v in meta.items():
            c.metadata[k] = v
        stream = None
        for i, img in enumerate(frames):
            if stream is None:
                h, w = img.shape[:2]
                stream = c.add_stream("libx264", rate=round(fps) or 30)
                stream.width, stream.height = w + w % 2, h + h % 2  # yuv420p needs even sizes
                stream.pix_fmt = "yuv420p"
                stream.codec_context.time_base = MP4_TIME_BASE
                stream.options = {"preset": "veryfast"}
            if img.shape[0] % 2 or img.shape[1] % 2:
                img = np.pad(img, ((0, img.shape[0] % 2), (0, img.shape[1] % 2), (0, 0)))
            f = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            f.pts = round(float(t[i] - t[0]) / MP4_TIME_BASE)
            f.time_base = MP4_TIME_BASE
            for p in stream.encode(f):
                c.mux(p)
        if stream is not None:
            for p in stream.encode():
                c.mux(p)


def _write_gif(frames: Iterator[np.ndarray], t: np.ndarray, out: Path, comment: str) -> None:
    from PIL import Image

    keep = gif_frame_indices(t)
    keep_set = set(keep.tolist())
    images = []
    for i, img in enumerate(frames):
        if i not in keep_set:
            continue
        im = Image.fromarray(img)
        if im.width > GIF_MAX_WIDTH:
            im = im.resize(
                (GIF_MAX_WIDTH, round(im.height * GIF_MAX_WIDTH / im.width)), Image.BILINEAR
            )
        images.append(im)
    if not images:
        raise ValueError("no frames to write")
    tk = t[keep[: len(images)]]
    durations = [round(1000 * d) for d in np.diff(tk)] + [round(1000 / GIF_FPS)]
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=durations, loop=0, comment=comment
    )


def render_overlay(
    seq: KeypointSeq,
    analysis: ClipAnalysis,
    out_path: str,
    video_path: str | None = None,
    gif: bool = False,
    max_seconds: float | None = None,
) -> Path:
    """Render the overlay to ``out_path`` (mp4, or GIF if ``gif``) and return the path.

    ``video_path`` is optional: without it the skeleton is drawn on a plain canvas sized from
    ``seq.size``. ``max_seconds`` keeps frames with ``t < max_seconds``.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(seq.t, dtype=np.float64)
    max_frames = int(np.sum(t < max_seconds)) if max_seconds is not None else None
    background = None
    if video_path is not None:
        from gaitlab.io.video import iter_frames

        background = (img for _, img in iter_frames(video_path))
    frames = render_frames(seq, analysis, background, max_frames=max_frames)
    t_used = t[:max_frames] if max_frames is not None else t
    label = "SYNTHETIC keypoints; never report" if seq.synthetic else f"gaitlab {seq.backend}"
    if gif:
        _write_gif(frames, t_used, out, comment=label)
    else:
        _write_mp4(frames, t_used, seq.fps, out, {"comment": label})
    return out
