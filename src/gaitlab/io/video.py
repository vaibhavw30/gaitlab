"""Video decoding with PyAV (TASKS M0, FR-1; decisions in ADR-0010).

- ``probe(path) -> VideoInfo``: fps (stream ``average_rate``), displayed size after rotation,
  exact frame count (demuxed packets), duration_s, codec, and ``vfr``.
- ``iter_frames(path)`` yields ``(t, rgb)`` in presentation order: ``t`` in s from the first
  presented frame's PTS (PTS × time_base, so ``t[0] == 0``), ``rgb`` uint8 (H, W, 3) upright.
- ``check_fps(info, cfg.capture)`` raises ``LowFpsError`` below ``min_fps`` unless allowed.

Timing always comes from container PTS, never ``frame / fps``: phone video can be VFR.
PyAV is imported lazily so the module imports without the ``video`` extra.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from gaitlab.config import CaptureCfg

VFR_REL_TOL = 0.01  # PTS deltas varying by more than 1 % of the median → VFR (beyond rounding)


class LowFpsError(ValueError):
    pass


@dataclass
class VideoInfo:
    """Container metadata for one clip.

    Attributes
    ----------
    path : str
    fps : float
        Nominal frame rate (stream ``average_rate``), frames/s.
    size : (int, int)
        (width, height) in px **after** applying rotation metadata.
    n_frames : int
        Number of video packets with a PTS (one per frame for phone codecs).
    duration_s : float
        Stream duration, s.
    codec : str
        Decoder name, e.g. ``"h264"``, ``"hevc"``.
    vfr : bool
        True if PTS deltas vary by more than ``VFR_REL_TOL`` of the median delta, beyond one
        tick of time_base rounding.
    """

    path: str
    fps: float
    size: tuple[int, int]
    n_frames: int
    duration_s: float
    codec: str
    vfr: bool


def _open(path: str) -> tuple[Any, Any]:
    import av

    container = av.open(str(path))
    if not container.streams.video:
        container.close()
        raise ValueError(f"{path}: no video stream")
    return container, container.streams.video[0]


def is_vfr(pts: np.ndarray, rel_tol: float = VFR_REL_TOL) -> bool:
    """Whether presentation timestamps describe a variable frame rate.

    Parameters
    ----------
    pts : (N,) int
        Presentation timestamps in ticks of the stream time_base, any order.
    rel_tol : float
        Allowed deviation of a delta from the median delta, as a fraction of the median.

    Returns
    -------
    bool
        True if any delta deviates from the median by more than
        ``max(rel_tol × median, 1 tick)``. The one-tick allowance absorbs rounding in coarse
        time bases (Matroska stores ms, so 120 fps gives 8 / 9 ms deltas).
    """
    d = np.diff(np.sort(np.asarray(pts, dtype=np.int64)))
    if d.size < 2:
        return False
    med = float(np.median(d))
    tol = max(rel_tol * med, 1.0)
    return bool(np.max(np.abs(d - med)) > tol)


def rotate_upright(img: np.ndarray, rotation: float) -> np.ndarray:
    """Apply display-matrix rotation so the image is shown as the camera intended.

    Parameters
    ----------
    img : (H, W, C) array
    rotation : float
        Counter-clockwise rotation in deg, as ``av.VideoFrame.rotation`` reports it (FFmpeg
        display matrix). Phones that tag ``rotate=90`` report −90 here. Rounded to the nearest
        multiple of 90; flips in the display matrix are ignored.

    Returns
    -------
    C-contiguous array, (W, H, C) for ±90°.
    """
    k = round(rotation / 90.0) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=k))


def probe(path: str) -> VideoInfo:
    """Read clip metadata. Demuxes every packet (no decode) and decodes one frame for rotation."""
    container, stream = _open(path)
    try:
        tb = stream.time_base
        pts = np.array(
            [p.pts for p in container.demux(stream) if p.pts is not None and p.size > 0],
            dtype=np.int64,
        )
    finally:
        container.close()
    if pts.size == 0:
        raise ValueError(f"{path}: no video frames")

    container, stream = _open(path)
    try:
        codec = stream.codec_context.name
        rate = stream.average_rate or stream.guessed_rate
        first = next(container.decode(stream), None)
        if first is None:
            raise ValueError(f"{path}: no decodable video frames")
        w, h = first.width, first.height
        if round(first.rotation / 90.0) % 2:
            w, h = h, w
        if stream.duration is not None:
            duration_s = float(stream.duration * tb)
        elif container.duration is not None:
            duration_s = container.duration / 1e6  # AV_TIME_BASE is µs
        else:
            duration_s = float((pts.max() - pts.min()) * tb)
    finally:
        container.close()

    if rate:
        fps = float(rate)
    else:
        fps = 1.0 / float(np.median(np.diff(np.sort(pts))) * tb) if pts.size > 1 else 0.0
    return VideoInfo(
        path=str(path),
        fps=fps,
        size=(int(w), int(h)),
        n_frames=int(pts.size),
        duration_s=duration_s,
        codec=codec,
        vfr=is_vfr(pts),
    )


def iter_frames(path: str) -> Iterator[tuple[float, np.ndarray]]:
    """Decode frames in presentation order.

    Yields
    ------
    t : float
        Seconds since the first presented frame, from PTS × time_base.
    rgb : (H, W, 3) uint8
        Upright RGB frame (rotation metadata applied), C-contiguous.
    """
    container, stream = _open(path)
    stream.thread_type = "AUTO"
    tb = stream.time_base
    try:
        pts0: int | None = None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"{path}: frame without PTS; cannot time it")
            if pts0 is None:
                pts0 = frame.pts
            t = float((frame.pts - pts0) * tb)
            yield t, rotate_upright(frame.to_ndarray(format="rgb24"), frame.rotation)
    finally:
        container.close()


def check_fps(info: VideoInfo, cfg: CaptureCfg, allow_low_fps: bool = False) -> None:
    """Reject clips below ``cfg.min_fps`` frames/s unless ``allow_low_fps`` (FR-1)."""
    if info.fps < cfg.min_fps and not allow_low_fps:
        raise LowFpsError(
            f"{info.path}: {info.fps:.1f} fps < {cfg.min_fps} (use --allow-low-fps for smoke tests)"
        )
