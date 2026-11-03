"""io.video: PyAV decode with PTS timestamps, rotation, VFR flag and fps check (TASKS M0, FR-1).

Clips are generated in-test with PyAV, so no data download is needed. Tests that need PyAV
are skipped when the ``video`` extra isn't installed (CI installs base + ml + dev only).
"""

from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from gaitlab.config import CaptureCfg
from gaitlab.io.video import LowFpsError, VideoInfo, check_fps, iter_frames, probe

requires_av = pytest.mark.skipif(
    importlib.util.find_spec("av") is None, reason="needs the 'video' extra (PyAV)"
)

W, H = 64, 48


def _gray(i: int, h: int = H, w: int = W) -> np.ndarray:
    """Solid gray frame whose level encodes the frame index (survives lossy coding)."""
    return np.full((h, w, 3), min(16 + 7 * i, 255), np.uint8)


def _marker(h: int = H, w: int = W) -> np.ndarray:
    """Black frame with a white 8×8 block in the top-left corner of the coded image."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:8, :8] = 255
    return img


def _write_clip(
    path: Path,
    frames: list[np.ndarray],
    fps: int = 120,
    pts: list[int] | None = None,
    time_base: Fraction | None = None,
    rotation: int | None = None,
) -> Path:
    """Encode ``frames`` (RGB uint8) with libx264. ``pts`` in units of ``time_base`` if given."""
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height = frames[0].shape[1], frames[0].shape[0]
        s.pix_fmt = "yuv420p"
        if time_base is not None:
            s.codec_context.time_base = time_base
        if rotation is not None:
            s.set_display_rotation(rotation)
        for i, img in enumerate(frames):
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            if pts is not None:
                f.pts = pts[i]
                f.time_base = time_base
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


@pytest.fixture
def cfr_clip(tmp_path: Path) -> Path:
    return _write_clip(tmp_path / "cfr.mp4", [_gray(i) for i in range(30)], fps=120)


@requires_av
def test_probe_reports_metadata(cfr_clip: Path):
    info = probe(str(cfr_clip))
    assert isinstance(info, VideoInfo)
    assert info.fps == pytest.approx(120.0)
    assert info.size == (W, H)
    assert info.n_frames == 30
    assert info.duration_s == pytest.approx(30 / 120, abs=1 / 120)
    assert info.codec == "h264"
    assert info.vfr is False


@requires_av
def test_iter_frames_pts_timestamps_shape_and_presentation_order(cfr_clip: Path):
    out = list(iter_frames(str(cfr_clip)))
    assert len(out) == 30
    t = np.array([ts for ts, _ in out])
    assert t[0] == 0.0
    np.testing.assert_allclose(np.diff(t), 1 / 120, atol=1e-9)
    for _, img in out:
        assert img.shape == (H, W, 3)
        assert img.dtype == np.uint8
    # libx264 uses B-frames, so decode order differs from presentation order; the gray level
    # encodes the frame index and must increase.
    levels = np.array([img.mean() for _, img in out])
    assert np.all(np.diff(levels) > 0)


@requires_av
def test_variable_frame_rate_flagged_and_timestamps_follow_pts(tmp_path: Path):
    tb = Fraction(1, 1000)
    deltas = [8, 8, 8, 12, 8, 8, 16, 8, 8, 8, 10, 8, 8, 8, 8, 8, 8, 8, 8]
    pts = [100, *list(100 + np.cumsum(deltas))]  # non-zero first pts: t is relative to it
    clip = _write_clip(
        tmp_path / "vfr.mp4", [_gray(i) for i in range(len(pts))], pts=pts, time_base=tb
    )
    info = probe(str(clip))
    assert info.vfr is True
    assert info.n_frames == len(pts)
    t = np.array([ts for ts, _ in iter_frames(str(clip))])
    np.testing.assert_allclose(t, (np.array(pts) - pts[0]) / 1000, atol=1e-9)


@requires_av
def test_coarse_timebase_rounding_is_not_vfr(tmp_path: Path):
    """Matroska stores ms timestamps: 120 fps → deltas of 8 or 9 ms. That's rounding, not VFR."""
    clip = _write_clip(tmp_path / "cfr.mkv", [_gray(i) for i in range(30)], fps=120)
    info = probe(str(clip))
    assert info.vfr is False
    assert info.n_frames == 30
    assert info.fps == pytest.approx(120.0)


# (display rotation written to the container, expected displayed size (W, H),
#  corner where the coded top-left marker ends up). Rotation is counter-clockwise, as in
#  FFmpeg's display matrix; a phone's "rotate=90" tag is -90 here (turn 90° clockwise to view).
ROTATIONS = [
    (-90, (H, W), "top_right"),
    (90, (H, W), "bottom_left"),
    (180, (W, H), "bottom_right"),
]


def _corner(img: np.ndarray, which: str) -> float:
    rows = slice(0, 8) if which.startswith("top") else slice(-8, None)
    cols = slice(0, 8) if which.endswith("left") else slice(-8, None)
    return float(img[rows, cols].mean())


@requires_av
@pytest.mark.parametrize(("rotation", "size", "corner"), ROTATIONS)
def test_rotation_metadata_is_applied(tmp_path: Path, rotation: int, size, corner: str):
    clip = _write_clip(tmp_path / "rot.mp4", [_marker() for _ in range(5)], rotation=rotation)
    info = probe(str(clip))
    assert info.size == size
    _, img = next(iter_frames(str(clip)))
    assert img.shape == (size[1], size[0], 3)
    assert img.flags["C_CONTIGUOUS"]
    assert _corner(img, corner) > 200
    others = {"top_left", "top_right", "bottom_left", "bottom_right"} - {corner}
    assert all(_corner(img, c) < 60 for c in others)


def _info(fps: float) -> VideoInfo:
    return VideoInfo(
        path="clip.mp4", fps=fps, size=(W, H), n_frames=10, duration_s=1.0, codec="h264", vfr=False
    )


def test_check_fps_rejects_low_fps_unless_allowed():
    cfg = CaptureCfg(min_fps=100.0)
    with pytest.raises(LowFpsError, match=r"clip\.mp4"):
        check_fps(_info(30.0), cfg)
    check_fps(_info(30.0), cfg, allow_low_fps=True)
    check_fps(_info(120.0), cfg)
    check_fps(_info(100.0), cfg)
