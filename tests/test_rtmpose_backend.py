"""pose.rtmpose_backend (TASKS M0): RTMPose Halpe-26 via rtmlib.

Unit tests inject a fake estimator, so they need neither rtmlib nor model files. The smoke test
runs the real model on the user's own clip and is marked ``data``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from gaitlab.pose.base import get_backend
from gaitlab.pose.rtmpose_backend import (
    N_KEYPOINTS,
    RTMPoseBackend,
    choose_person,
    ort_device,
)
from gaitlab.pose.skeleton import CANONICAL, HALPE_26, J
from gaitlab.types import KeypointSeq

W, H = 80, 60


def _person(i: int, offset: float = 0.0, score: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """26 keypoints in pixels: keypoint k at (k + offset, k + i); every score = ``score``."""
    k = np.arange(N_KEYPOINTS, dtype=np.float32)
    return np.stack([k + offset, k + i], axis=1), np.full(N_KEYPOINTS, score, np.float32)


class FakeEstimator:
    """Stands in for rtmlib's detector + pose model. ``people(i)`` lists frame i's persons."""

    def __init__(self, people=None, k: int = N_KEYPOINTS) -> None:
        self.people = people or (lambda i: [_person(i)])
        self.k = k
        self.seen: list[np.ndarray] = []
        self.opened: list[tuple[str, str, str]] = []

    @contextmanager
    def __call__(self, mode: str, device: str, model_dir: str) -> Iterator[tuple[object, dict]]:
        self.opened.append((mode, device, model_dir))
        i = -1

        def estimate(bgr: np.ndarray):
            nonlocal i
            i += 1
            self.seen.append(bgr)
            found = self.people(i)
            if not found:
                return None
            kpts = np.stack([p[0][: self.k] for p in found])
            scores = np.stack([p[1][: self.k] for p in found])
            return kpts, scores

        yield estimate, {"rtmlib": "fake", "pose_model": "fake-halpe26.onnx"}


def _frames(n: int, fps: float = 120.0) -> list[tuple[float, np.ndarray]]:
    frames = []
    for i in range(n):
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[..., 0] = 200  # red only: BGR conversion puts it in the last channel
        frames.append((i / fps, rgb))
    return frames


def _backend(est: FakeEstimator, **kw) -> RTMPoseBackend:
    return RTMPoseBackend(open_estimator=est, device="cpu", **kw)


def test_run_maps_halpe26_to_canonical_pixels():
    est = FakeEstimator()
    seq = _backend(est).run(_frames(4), 120.0, (W, H))
    assert isinstance(seq, KeypointSeq) and seq.backend == "rtmpose"
    assert seq.xy.shape == (4, len(CANONICAL), 2) and seq.xy.dtype == np.float32
    assert seq.vis.dtype == np.float32
    np.testing.assert_allclose(seq.t, np.arange(4) / 120.0)
    assert seq.fps == 120.0 and seq.size == (W, H)
    for name in ("l_heel", "r_toe", "nose", "l_knee"):
        k = HALPE_26[name]
        np.testing.assert_allclose(seq.xy[2, J[name]], [k, k + 2])  # already pixels
    assert seq.meta["rtmlib"] == "fake" and seq.meta["mode"] == "performance"
    assert est.opened == [("performance", "cpu", "models/rtmlib")]


def test_frames_go_to_rtmlib_as_contiguous_bgr():
    est = FakeEstimator()
    _backend(est).run(_frames(2), 120.0, (W, H))
    bgr = est.seen[0]
    assert bgr.flags["C_CONTIGUOUS"]
    assert bgr[0, 0].tolist() == [0, 0, 200]


def test_no_detection_is_nan_with_zero_visibility():
    """rtmlib's pose model falls back to the whole image when given no boxes, so the backend
    must not call it: an empty frame has no person, not a confident full-frame skeleton."""
    est = FakeEstimator(people=lambda i: [] if i in (1, 3) else [_person(i)])
    seq = _backend(est).run(_frames(5), 120.0, (W, H))
    for i in (1, 3):
        assert np.isnan(seq.xy[i]).all() and (seq.vis[i] == 0).all()
    assert np.isfinite(seq.xy[[0, 2, 4]]).all()
    assert seq.meta["frames_without_person"] == 2


def test_the_person_with_the_highest_mean_score_is_kept():
    est = FakeEstimator(people=lambda i: [_person(i, 100.0, 0.4), _person(i, 0.0, 0.8)])
    seq = _backend(est).run(_frames(3), 120.0, (W, H))
    assert seq.xy[0, J["nose"], 0] == 0.0  # the offset-0 person (higher score)
    assert seq.meta["frames_with_several_people"] == 3
    kpts = np.stack([_person(0, 1.0, 0.2)[0], _person(0, 2.0, 0.7)[0]])
    scores = np.stack([np.full(26, 0.2), np.full(26, 0.7)])
    assert choose_person(kpts, scores) == 1


def test_scores_are_clipped_to_unit_visibility_and_summarized_in_meta():
    est = FakeEstimator(people=lambda i: [_person(i, score=1.3 if i == 0 else 0.6)])
    seq = _backend(est).run(_frames(3), 120.0, (W, H))
    assert seq.vis.max() == 1.0 and seq.vis.min() >= 0.0
    q = seq.meta["score_quantiles"]
    assert set(q) == {"p05", "p50", "p95"} and q["p95"] > 1.0  # raw scores, before clipping


def test_a_model_without_feet_is_refused():
    with pytest.raises(ValueError, match="Halpe-26"):
        _backend(FakeEstimator(k=17)).run(_frames(2), 120.0, (W, H))


def test_frame_size_mismatch_and_empty_clip_raise():
    with pytest.raises(ValueError, match="size"):
        _backend(FakeEstimator()).run(_frames(2), 120.0, (H, W))
    with pytest.raises(ValueError, match="no frames"):
        _backend(FakeEstimator()).run([], 120.0, (W, H))


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        RTMPoseBackend(mode="fastest")


def test_device_is_cuda_only_when_onnxruntime_has_it():
    assert ort_device(["CUDAExecutionProvider", "CPUExecutionProvider"], None) == "cuda"
    assert ort_device(["CoreMLExecutionProvider", "CPUExecutionProvider"], None) == "cpu"
    assert ort_device(["CPUExecutionProvider"], "cpu") == "cpu"
    with pytest.raises(ValueError, match="cuda"):
        ort_device(["CPUExecutionProvider"], "cuda")


def test_factory_builds_the_rtmpose_backend():
    b = get_backend("rtmpose", mode="lightweight", device="cpu")
    assert isinstance(b, RTMPoseBackend) and b.mode == "lightweight"


def test_module_does_not_import_rtmlib_onnxruntime_or_cv2_at_import_time():
    code = (
        "import sys, gaitlab.pose.rtmpose_backend; "
        "bad = {'rtmlib', 'onnxruntime', 'cv2', 'torch'} & set(sys.modules); assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(importlib.util.find_spec("rtmlib") is None, reason="needs the 'rtm' extra")
def test_every_mode_uses_a_halpe26_pose_model():
    from gaitlab.pose.rtmpose_backend import model_urls

    for mode in ("performance", "balanced", "lightweight"):
        det, pose = model_urls(mode)
        assert "halpe26" in pose["url"] and det["url"].endswith(".zip")


# --- smoke test on the user's own clip --------------------------------------------------------

SMOKE_CLIP = Path("data/raw/SMOKE/natural.mp4")
SMOKE_SECONDS = 3.0


@pytest.mark.data
def test_smoke_own_clip_rtmpose():
    if importlib.util.find_spec("rtmlib") is None:
        pytest.skip("needs the 'rtm' extra")
    if not SMOKE_CLIP.exists():
        pytest.skip(f"put your own smoke clip at {SMOKE_CLIP}")
    from gaitlab.io.video import iter_frames, probe

    info = probe(str(SMOKE_CLIP))
    n = min(info.n_frames, int(SMOKE_SECONDS * info.fps))
    seq = RTMPoseBackend().run(islice(iter_frames(str(SMOKE_CLIP)), n), info.fps, info.size)

    assert seq.n_frames == n and np.all(np.diff(seq.t) > 0)
    detected = np.isfinite(seq.xy).all(axis=(1, 2))
    assert detected.mean() > 0.9, f"person detected in {detected.mean():.0%} of frames"
    d = seq.xy[detected]
    for side in ("l", "r"):  # upright, y down
        assert np.mean(d[:, J[f"{side}_hip"], 1] < d[:, J[f"{side}_ankle"], 1]) > 0.9
        assert np.mean(d[:, J[f"{side}_shoulder"], 1] < d[:, J[f"{side}_hip"], 1]) > 0.9
        assert np.isfinite(d[:, J[f"{side}_heel"]]).all()  # Halpe-26 has feet
