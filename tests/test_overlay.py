"""viz.overlay: skeleton, IC/TO markers and near-side angles, with or without a video (TASKS M0)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gaitlab.config import PipelineConfig
from gaitlab.data.synthetic import synth_clip
from gaitlab.gait.features import analyze_clip, knee_flexion_deg, shin_angle_deg

requires_pil = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="needs the 'video' extra (Pillow)"
)
requires_av = pytest.mark.skipif(
    importlib.util.find_spec("av") is None, reason="needs the 'video' extra (PyAV)"
)

W, H = 400, 300


@pytest.fixture(scope="module")
def clip():
    seq, _ = synth_clip(duration_s=3.0, size=(W, H), px_per_m=100.0)
    analysis = analyze_clip(seq, PipelineConfig())
    assert len(analysis.events.ic) >= 2, "fixture needs IC events to check markers"
    return seq, analysis


def test_annotations_mark_ic_and_to_on_event_frames(clip):
    from gaitlab.viz.overlay import annotate

    seq, an = clip
    ann = annotate(seq, an)
    assert len(ann) == seq.n_frames
    assert [a.frame for a in ann] == list(range(seq.n_frames))
    assert sorted(a.frame for a in ann if a.ic_age == 0) == an.events.ic.tolist()
    assert sorted(a.frame for a in ann if a.to_age == 0) == an.events.to.tolist()
    # Markers are held for a short time after the event, never shown before it.
    first_ic = int(an.events.ic[0])
    assert all(a.ic_age is None for a in ann[:first_ic])
    assert ann[first_ic + 1].ic_age == 1


def test_annotation_angles_are_the_near_side_feature_angles(clip):
    from gaitlab.viz.overlay import annotate

    seq, an = clip
    ann = annotate(seq, an)
    knee = knee_flexion_deg(an.xy_up, an.side)
    shin = shin_angle_deg(an.xy_up, an.side, an.d)
    np.testing.assert_allclose([a.knee_flex_deg for a in ann], knee)
    np.testing.assert_allclose([a.shin_angle_deg for a in ann], shin)
    # The per-frame shin angle at IC is the stride feature.
    for s in an.strides:
        assert ann[s.ic_frame].shin_angle_deg == pytest.approx(s.shin_angle_deg)
        assert ann[s.ic_frame].cadence_spm == pytest.approx(s.cadence_spm)


@requires_pil
def test_render_frames_one_canvas_frame_per_keypoint_frame(clip):
    from gaitlab.viz.overlay import render_frames

    seq, an = clip
    frames = list(render_frames(seq, an))
    assert len(frames) == seq.n_frames
    for f in frames:
        assert f.shape == (H, W, 3) and f.dtype == np.uint8


def _ic_pixels(img: np.ndarray) -> np.ndarray:
    from gaitlab.viz.overlay import IC_COLOR

    return np.argwhere((img == np.array(IC_COLOR, np.uint8)).all(axis=-1))


@requires_pil
def test_ic_marker_drawn_at_the_heel_on_ic_frames_only(clip):
    from gaitlab.pose.skeleton import J
    from gaitlab.viz.overlay import annotate, render_frames

    seq, an = clip
    frames = list(render_frames(seq, an))
    ann = annotate(seq, an)
    heel = J[f"{an.side}_heel"]
    for ic in an.events.ic:
        px = _ic_pixels(frames[ic])
        assert len(px) > 0, f"no IC marker on IC frame {ic}"
        hx, hy = seq.xy[ic, heel]
        dist = np.hypot(px[:, 1] - hx, px[:, 0] - hy)
        assert dist.max() < 20, "IC marker should ring the near-side heel"
    for a, f in zip(ann, frames, strict=True):
        if a.ic_age is None:
            assert len(_ic_pixels(f)) == 0, f"IC marker on frame {a.frame} outside hold"


@requires_pil
def test_synthetic_clips_are_watermarked(clip):
    from gaitlab.viz.overlay import render_frames

    seq, an = clip
    real = type(seq)(seq.xy, seq.vis, seq.t, seq.fps, seq.size, "mediapipe", {})
    a = next(iter(render_frames(seq, an)))
    b = next(iter(render_frames(real, an)))
    assert not np.array_equal(a, b)


@requires_pil
def test_video_background_must_match_length_and_size(clip):
    from gaitlab.viz.overlay import render_frames

    seq, an = clip
    bg = [np.full((H, W, 3), 200, np.uint8) for _ in range(seq.n_frames)]
    frames = list(render_frames(seq, an, background=iter(bg)))
    assert len(frames) == seq.n_frames
    assert frames[0][0, 0].tolist() == [200, 200, 200]  # background shows through
    with pytest.raises(ValueError, match="frames"):
        list(render_frames(seq, an, background=iter(bg[:-5])))
    with pytest.raises(ValueError, match="frames"):
        list(render_frames(seq, an, background=iter([*bg, bg[0]])))
    with pytest.raises(ValueError, match="size"):
        list(render_frames(seq, an, background=iter([np.zeros((H, W + 2, 3), np.uint8)])))


def test_gif_frame_indices_sample_by_timestamps():
    from gaitlab.viz.overlay import gif_frame_indices

    t = np.arange(240) / 120.0  # 2 s at 120 fps
    idx = gif_frame_indices(t, gif_fps=15.0)
    assert len(idx) == 30
    assert idx[0] == 0
    np.testing.assert_allclose(np.diff(t[idx]), 1 / 15, atol=1 / 120)


@requires_pil
def test_render_overlay_gif_downscaled(clip, tmp_path: Path):
    from PIL import Image

    from gaitlab.viz.overlay import gif_frame_indices, render_overlay

    seq, an = clip
    big = type(seq)(seq.xy * 2, seq.vis, seq.t, seq.fps, (W * 2, H * 2), seq.backend, seq.meta)
    out = render_overlay(big, an, str(tmp_path / "o.gif"), gif=True)
    with Image.open(out) as im:
        assert im.size == (480, 360)
        assert im.n_frames == len(gif_frame_indices(seq.t))


@requires_av
@requires_pil
def test_render_overlay_mp4_has_n_frames_and_pts_timing(clip, tmp_path: Path):
    from gaitlab.io.video import iter_frames, probe
    from gaitlab.viz.overlay import render_overlay

    seq, an = clip
    out = render_overlay(seq, an, str(tmp_path / "o.mp4"))
    info = probe(str(out))
    assert info.n_frames == seq.n_frames
    assert info.size == (W, H)
    t = np.array([ts for ts, _ in iter_frames(str(out))])
    np.testing.assert_allclose(t, seq.t, atol=1e-3)


@requires_av
@requires_pil
def test_render_overlay_on_a_video(clip, tmp_path: Path):
    import av

    from gaitlab.io.video import probe
    from gaitlab.viz.overlay import render_overlay

    seq, an = clip
    src = tmp_path / "src.mp4"
    with av.open(str(src), "w") as c:
        s = c.add_stream("libx264", rate=120)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        for _ in range(seq.n_frames):
            f = av.VideoFrame.from_ndarray(np.full((H, W, 3), 90, np.uint8), format="rgb24")
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    out = render_overlay(seq, an, str(tmp_path / "o.mp4"), video_path=str(src), max_seconds=1.0)
    assert probe(str(out)).n_frames == int(np.sum(seq.t < 1.0))
