"""Shared dataclasses. Schemas mirror docs/TECHNICAL_PLAN.md §4."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaitlab.pose.skeleton import CANONICAL


@dataclass
class KeypointSeq:
    """A clip's keypoints in the canonical skeleton.

    Attributes
    ----------
    xy : (T, J, 2) float32, pixels, origin top-left, **y down** (as stored).
    vis : (T, J) float32 in [0, 1].
    t : (T,) float64 seconds from clip start (container PTS, not frame/fps).
    fps : nominal frames per second.
    size : (width, height) in pixels.
    backend : "mediapipe" | "rtmpose" | "mocap_proj" | "synthetic" | ...
    meta : free-form metadata (backend version, model hash, ...).
    """

    xy: np.ndarray
    vis: np.ndarray
    t: np.ndarray
    fps: float
    size: tuple[int, int]
    backend: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_t, n_j = self.xy.shape[:2]
        if n_j != len(CANONICAL):
            raise ValueError(f"expected {len(CANONICAL)} joints, got {n_j}")
        if self.xy.shape != (n_t, n_j, 2):
            raise ValueError(f"xy must be (T, J, 2), got {self.xy.shape}")
        if self.vis.shape != (n_t, n_j):
            raise ValueError(f"vis must be (T, J), got {self.vis.shape}")
        if self.t.shape != (n_t,):
            raise ValueError(f"t must be (T,), got {self.t.shape}")

    @property
    def n_frames(self) -> int:
        return int(self.xy.shape[0])

    def save(self, path: str) -> None:
        import json

        np.savez_compressed(
            path,
            xy=self.xy.astype(np.float32),
            vis=self.vis.astype(np.float32),
            t=self.t.astype(np.float64),
            fps=np.float64(self.fps),
            size=np.asarray(self.size, dtype=np.int32),
            backend=np.asarray(self.backend),
            meta=np.asarray(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str) -> KeypointSeq:
        import json

        z = np.load(path, allow_pickle=False)
        return cls(
            xy=z["xy"],
            vis=z["vis"],
            t=z["t"],
            fps=float(z["fps"]),
            size=(int(z["size"][0]), int(z["size"][1])),
            backend=str(z["backend"]),
            meta=json.loads(str(z["meta"])),
        )


@dataclass
class Events:
    """Gait events for one foot. Frame indices into the clip."""

    side: str  # "l" | "r"
    ic: np.ndarray  # initial-contact frames, int, ascending
    to: np.ndarray  # toe-off frames, int, same length as ic (to[k] belongs to ic[k])


@dataclass
class Stride:
    """One near-side stride [ic, next_ic). Units in field names; L = leg lengths."""

    stride_idx: int
    side: str
    ic_frame: int
    to_frame: int
    next_ic_frame: int
    t_ic: float
    stride_time_s: float
    contact_time_s: float
    duty_factor: float
    cadence_spm: float
    overstride_L: float
    shin_angle_deg: float
    knee_flex_ic_deg: float
    knee_flex_peak_deg: float
    foot_strike_deg: float
    foot_strike_calibrated: bool
    vert_osc_L: float
    trunk_lean_deg: float
    quality: float


FEATURE_FIELDS: tuple[str, ...] = (
    "cadence_spm",
    "overstride_L",
    "shin_angle_deg",
    "knee_flex_ic_deg",
    "knee_flex_peak_deg",
    "foot_strike_deg",
    "vert_osc_L",
    "trunk_lean_deg",
    "contact_time_s",
    "duty_factor",
)
"""Per-stride features summarized into windows (median and IQR each)."""


@dataclass
class Window:
    """A run of consecutive valid strides with per-feature median and IQR."""

    window_id: str
    session_id: str
    clip_id: str
    start_stride: int
    n_strides: int
    t_start: float
    t_end: float
    quality: float
    med: dict[str, float]
    iqr: dict[str, float]
    condition: str = ""
    speed_kmh: float | None = None

    def flat(self) -> dict[str, float | str | int | None]:
        """Flatten to the windows.parquet row schema (TECHNICAL_PLAN §4.4)."""
        row: dict[str, float | str | int | None] = {
            "window_id": self.window_id,
            "session_id": self.session_id,
            "clip_id": self.clip_id,
            "condition": self.condition,
            "start_stride": self.start_stride,
            "n_strides": self.n_strides,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "speed_kmh": self.speed_kmh,
            "quality": self.quality,
        }
        for k, v in self.med.items():
            row[f"{k}_med"] = v
        for k, v in self.iqr.items():
            row[f"{k}_iqr"] = v
        return row
