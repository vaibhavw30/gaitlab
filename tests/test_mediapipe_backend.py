"""pose.mediapipe_backend (TASKS M0, FR-2).

Unit tests inject a fake detector, so they need neither mediapipe nor the model file. The smoke
test runs the real heavy model on the user's own clip and is marked ``data``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from gaitlab.pose.mediapipe_backend import MediaPipeBackend, video_timestamps_ms
from gaitlab.pose.skeleton import CANONICAL, MEDIAPIPE_33, J
from gaitlab.types import KeypointSeq

W, H = 80, 60


def test_timestamps_ms_follow_pts_and_strictly_increase():
    t = np.array([0.0, 1 / 240, 2 / 240, 4 / 240])
    assert video_timestamps_ms(t) == [0, 4, 8, 17]
    # Ties after rounding (sub-ms frame spacing, duplicate PTS) are bumped by 1 ms.
    assert video_timestamps_ms(np.array([0.0, 0.0002, 0.0004, 0.010])) == [0, 1, 2, 10]
    assert video_timestamps_ms(np.array([0.0, 0.0, 0.0004, 0.0021])) == [0, 1, 2, 3]


def _fake_landmarks(i: int) -> tuple[np.ndarray, np.ndarray]:
    """33 normalized landmarks: landmark k at x = k/40, y = (k + i)/80; visibility k/33."""
    k = np.arange(33, dtype=np.float32)
    xy = np.stack([k / 40, (k + i) / 80], axis=1)
    return xy, k / 33


class FakeDetector:
    """Stands in for the Tasks API landmarker. Frames listed in ``missing`` have no person."""

    def __init__(self, missing: set[int] = frozenset()) -> None:
        self.missing = set(missing)
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.opened = 0

    @contextmanager
    def __call__(self, model_path: str) -> Iterator[tuple[object, dict]]:
        self.opened += 1
        i = -1

        def detect(rgb: np.ndarray, ts_ms: int):
            nonlocal i
            i += 1
            self.calls.append((rgb.shape, ts_ms))
            return None if i in self.missing else _fake_landmarks(i)

        yield detect, {"mediapipe": "fake", "model_sha256": "0" * 64}


def _frames(n: int, fps: float = 120.0) -> list[tuple[float, np.ndarray]]:
    return [(i / fps, np.zeros((H, W, 3), np.uint8)) for i in range(n)]


def test_run_returns_canonical_pixels_and_passes_timestamps():
    det = FakeDetector()
    seq = MediaPipeBackend(model_path="m.task", open_detector=det).run(_frames(5), 120.0, (W, H))
    assert isinstance(seq, KeypointSeq)
    assert seq.backend == "mediapipe"
    assert seq.xy.shape == (5, len(CANONICAL), 2)
    assert seq.xy.dtype == np.float32 and seq.vis.dtype == np.float32
    assert seq.fps == 120.0 and seq.size == (W, H)
    np.testing.assert_allclose(seq.t, np.arange(5) / 120.0)
    assert [ts for _, ts in det.calls] == video_timestamps_ms(seq.t)
    assert det.opened == 1
    # Normalized → pixels (x · width, y · height), mapped 33 → canonical by name.
    for name in ("l_heel", "r_toe", "nose", "r_knee"):
        k = MEDIAPIPE_33[name]
        xy, vis = _fake_landmarks(2)
        np.testing.assert_allclose(seq.xy[2, J[name]], [xy[k, 0] * W, xy[k, 1] * H], rtol=1e-6)
        assert seq.vis[2, J[name]] == pytest.approx(vis[k])
    assert seq.meta["mediapipe"] == "fake"
    assert len(seq.meta["model_sha256"]) == 64


def test_frames_without_a_person_are_nan_with_zero_visibility():
    seq = MediaPipeBackend(open_detector=FakeDetector(missing={1, 3})).run(
        _frames(5), 120.0, (W, H)
    )
    for i in (1, 3):
        assert np.isnan(seq.xy[i]).all()
        assert (seq.vis[i] == 0).all()
    assert np.isfinite(seq.xy[[0, 2, 4]]).all()


def test_frame_size_mismatch_raises():
    with pytest.raises(ValueError, match="size"):
        MediaPipeBackend(open_detector=FakeDetector()).run(_frames(2), 120.0, (H, W))


def test_empty_clip_raises():
    with pytest.raises(ValueError, match="no frames"):
        MediaPipeBackend(open_detector=FakeDetector()).run([], 120.0, (W, H))


def test_module_does_not_import_mediapipe_at_import_time():
    """CPU paths must not pull in mediapipe at module level (TASKS: where work runs)."""
    import subprocess
    import sys

    code = (
        "import sys, gaitlab.pose.mediapipe_backend, gaitlab.pose.base; "
        "assert 'mediapipe' not in sys.modules, 'mediapipe imported'; "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- smoke test on the user's own clip --------------------------------------------------------

SMOKE_CLIP = Path("data/raw/SMOKE/natural.mp4")
MODEL = Path("models/pose_landmarker_heavy.task")
SMOKE_SECONDS = 3.0


@pytest.mark.data
def test_smoke_own_clip_heavy_model():
    if importlib.util.find_spec("mediapipe") is None:
        pytest.skip("needs the 'mediapipe' extra")
    if not SMOKE_CLIP.exists():
        pytest.skip(f"put your own smoke clip at {SMOKE_CLIP}")
    if not MODEL.exists():
        pytest.skip(f"run scripts/download_models.sh to fetch {MODEL}")
    from gaitlab.io.video import iter_frames, probe

    info = probe(str(SMOKE_CLIP))
    n = min(info.n_frames, int(SMOKE_SECONDS * info.fps))
    frames = islice(iter_frames(str(SMOKE_CLIP)), n)
    seq = MediaPipeBackend(model_path=str(MODEL)).run(frames, info.fps, info.size)

    assert seq.n_frames == n
    assert np.all(np.diff(seq.t) > 0)
    assert len(seq.meta["model_sha256"]) == 64 and seq.meta["mediapipe"]
    detected = np.isfinite(seq.xy).all(axis=(1, 2))
    # A runner fills the side view for the whole clip: nearly every frame has a detection.
    assert detected.mean() > 0.9, f"person detected in {detected.mean():.0%} of frames"
    d = seq.xy[detected]
    # Upright after rotation (y down): hips above ankles, shoulders above hips.
    for side in ("l", "r"):
        assert np.mean(d[:, J[f"{side}_hip"], 1] < d[:, J[f"{side}_ankle"], 1]) > 0.9
        assert np.mean(d[:, J[f"{side}_shoulder"], 1] < d[:, J[f"{side}_hip"], 1]) > 0.9
    assert ((seq.vis >= 0) & (seq.vis <= 1)).all()
