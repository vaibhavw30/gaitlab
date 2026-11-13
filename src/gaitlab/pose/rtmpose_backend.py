"""RTMPose backend via rtmlib (TASKS M0). Second backend; the primary is chosen by ADR-0005.

- Models: rtmlib's ``BodyWithFeet`` model zoo, a YOLOX person detector plus an RTMPose
  **Halpe-26** (body7) pose model, so heel and big toe exist. ``mode`` (config
  ``pose.rtmpose_mode``) picks the pair: ``performance`` | ``balanced`` | ``lightweight``.
  COCO-17 models have no feet and are refused (``N_KEYPOINTS``).
- Model files are downloaded with rtmlib's own ``download_checkpoint`` into ``models/rtmlib``
  inside the repo (git-ignored; CLAUDE.md rule 8), not rtmlib's default ``~/.cache``.
- Detector and pose model are called separately. rtmlib's pose model falls back to the whole
  image when given no boxes, which would turn an empty frame into a confident skeleton; here a
  frame without a detected person is NaN xy with vis 0.
- rtmlib expects BGR (OpenCV convention): RGB frames are flipped with ``rgb[..., ::-1]``.
- Keypoints come back in pixels. If several people are detected, the one with the highest mean
  keypoint score is kept (single runner). Mapped with ``skeleton.to_canonical(..., "halpe26")``.
- Scores are SimCC confidences, not calibrated visibilities, and can exceed 1. ``vis`` is the
  score clipped to [0, 1]; ``meta["score_quantiles"]`` keeps the raw p05/p50/p95 so
  ``clean.visibility_min`` can be tuned per backend.
- Device: onnxruntime ``cuda`` when its CUDA provider is present (the GPU host), else ``cpu``.

rtmlib, onnxruntime and cv2 are imported only inside ``model_urls`` and ``rtmlib_estimator``,
so this module imports on installs without the ``rtm`` extra.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import numpy as np

from gaitlab.pose.mediapipe_backend import file_sha256
from gaitlab.pose.skeleton import to_canonical
from gaitlab.types import KeypointSeq

N_KEYPOINTS = 26  # Halpe-26
MODES = ("performance", "balanced", "lightweight")
MODEL_DIR = "models/rtmlib"

People = tuple[np.ndarray, np.ndarray]
"""(N, 26, 2) keypoints in pixels and (N, 26) scores, one row per detected person."""
EstimateFn = Callable[[np.ndarray], People | None]
"""BGR uint8 (H, W, 3) → people, or None if the detector found nobody."""
OpenEstimator = Callable[[str, str | None, str], AbstractContextManager[tuple[EstimateFn, dict]]]
"""(mode, device, model_dir) → context manager yielding (estimate function, backend meta)."""


def ort_device(providers: Sequence[str], prefer: str | None) -> str:
    """``cuda`` if onnxruntime has its CUDA provider (or it was asked for), else ``cpu``."""
    has_cuda = "CUDAExecutionProvider" in providers
    if prefer is None:
        return "cuda" if has_cuda else "cpu"
    if prefer == "cuda" and not has_cuda:
        raise ValueError("device 'cuda' requested but onnxruntime has no CUDAExecutionProvider")
    if prefer not in ("cuda", "cpu"):
        raise ValueError(f"device must be 'cuda' or 'cpu', got {prefer!r}")
    return prefer


def choose_person(keypoints: np.ndarray, scores: np.ndarray) -> int:
    """Index of the person with the highest mean keypoint score."""
    return int(np.argmax(np.asarray(scores, dtype=np.float64).mean(axis=1)))


def model_urls(mode: str) -> tuple[dict, dict]:
    """(detector, pose) as ``{"url", "input_size"}`` from rtmlib's ``BodyWithFeet`` zoo."""
    from rtmlib import BodyWithFeet

    if mode not in BodyWithFeet.MODE:
        raise ValueError(f"mode must be one of {sorted(BodyWithFeet.MODE)}, got {mode!r}")
    m = BodyWithFeet.MODE[mode]
    return (
        {"url": m["det"], "input_size": tuple(m["det_input_size"])},
        {"url": m["pose"], "input_size": tuple(m["pose_input_size"])},
    )


@contextmanager
def rtmlib_estimator(
    mode: str, device: str | None, model_dir: str
) -> Iterator[tuple[EstimateFn, dict]]:
    """YOLOX person detector + RTMPose Halpe-26, models cached under ``model_dir``."""
    import onnxruntime
    import rtmlib
    from rtmlib import YOLOX, RTMPose
    from rtmlib.tools.file import download_checkpoint

    det_spec, pose_spec = model_urls(mode)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    det_path = download_checkpoint(det_spec["url"], dst_dir=model_dir)
    pose_path = download_checkpoint(pose_spec["url"], dst_dir=model_dir)
    dev = ort_device(onnxruntime.get_available_providers(), device)
    det = YOLOX(det_path, model_input_size=det_spec["input_size"], backend="onnxruntime",
                device=dev)  # fmt: skip
    pose = RTMPose(pose_path, model_input_size=pose_spec["input_size"], to_openpose=False,
                   backend="onnxruntime", device=dev)  # fmt: skip
    meta = {
        "rtmlib": rtmlib.__version__ if hasattr(rtmlib, "__version__") else _version("rtmlib"),
        "onnxruntime": onnxruntime.__version__,
        "device": dev,
        "det_model": Path(det_path).name,
        "det_sha256": file_sha256(det_path),
        "pose_model": Path(pose_path).name,
        "pose_sha256": file_sha256(pose_path),
    }

    def estimate(bgr: np.ndarray) -> People | None:
        boxes = det(bgr)
        if len(boxes) == 0:
            return None
        kpts, scores = pose(bgr, bboxes=boxes)
        return np.asarray(kpts), np.asarray(scores)

    yield estimate, meta


def _version(pkg: str) -> str:
    import importlib.metadata as md

    return md.version(pkg)


class RTMPoseBackend:
    """``PoseBackend`` running RTMPose Halpe-26 through rtmlib.

    Parameters
    ----------
    mode : str
        rtmlib ``BodyWithFeet`` mode (``performance`` | ``balanced`` | ``lightweight``).
    device : str, optional
        ``cuda`` or ``cpu``; default: ``cuda`` when onnxruntime has it, else ``cpu``.
    model_dir : str
        Where the ONNX models are downloaded and cached (inside the repo).
    open_estimator : callable, optional
        Seam for tests: (mode, device, model_dir) → context manager yielding
        ``(estimate, meta)``. Defaults to ``rtmlib_estimator``.
    """

    name = "rtmpose"

    def __init__(
        self,
        mode: str = "performance",
        device: str | None = None,
        model_dir: str = MODEL_DIR,
        open_estimator: OpenEstimator | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.device = device
        self.model_dir = model_dir
        self._open = open_estimator or rtmlib_estimator

    def run(
        self,
        frames: Iterable[tuple[float, np.ndarray]],
        fps: float,
        size: tuple[int, int],
    ) -> KeypointSeq:
        """Estimate keypoints for every frame; see ``pose.base.PoseBackend.run``."""
        width, height = size
        ts: list[float] = []
        xy_rows: list[np.ndarray] = []
        score_rows: list[np.ndarray] = []
        raw_scores: list[np.ndarray] = []
        n_empty = n_several = 0
        with self._open(self.mode, self.device, self.model_dir) as (estimate, meta):
            for t, rgb in frames:
                if rgb.shape[:2] != (height, width):
                    raise ValueError(
                        f"frame size {rgb.shape[1]}x{rgb.shape[0]} != clip size {width}x{height}"
                    )
                found = estimate(np.ascontiguousarray(rgb[..., ::-1]))
                if found is None or len(found[0]) == 0:
                    n_empty += 1
                    xy = np.full((N_KEYPOINTS, 2), np.nan, dtype=np.float32)
                    score = np.zeros(N_KEYPOINTS, dtype=np.float32)
                else:
                    kpts, scores = found
                    if kpts.shape[1:] != (N_KEYPOINTS, 2):
                        raise ValueError(
                            f"pose model gave {kpts.shape[1]} keypoints; need Halpe-26 "
                            f"({N_KEYPOINTS}) for heel and toe"
                        )
                    n_several += len(kpts) > 1
                    p = choose_person(kpts, scores)
                    xy = kpts[p].astype(np.float32)
                    score = scores[p].astype(np.float32)
                    raw_scores.append(score)
                ts.append(float(t))
                xy_rows.append(xy)
                score_rows.append(score)
        if not ts:
            raise ValueError("no frames to run pose estimation on")
        vis = np.clip(np.stack(score_rows), 0.0, 1.0)
        cxy, cvis = to_canonical(np.stack(xy_rows), vis, "halpe26")
        raw = np.concatenate(raw_scores) if raw_scores else np.zeros(0)
        quantiles = (
            {f"p{q:02d}": float(np.percentile(raw, q)) for q in (5, 50, 95)} if raw.size else {}
        )
        return KeypointSeq(
            xy=cxy,
            vis=cvis,
            t=np.asarray(ts, dtype=np.float64),
            fps=float(fps),
            size=(int(width), int(height)),
            backend=self.name,
            meta={
                **meta,
                "mode": self.mode,
                "frames_without_person": n_empty,
                "frames_with_several_people": int(n_several),
                "score_quantiles": quantiles,
            },
        )
