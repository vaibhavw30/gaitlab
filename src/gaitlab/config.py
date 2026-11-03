"""Typed configuration. YAML in configs/ → pydantic models → a stable hash for provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaptureCfg(_Strict):
    min_fps: float = 100.0
    contact_time_min_fps: float = 200.0


class PoseCfg(_Strict):
    backend: Literal["mediapipe", "rtmpose"] = "mediapipe"
    mediapipe_model: str = "models/pose_landmarker_heavy.task"
    rtmpose_mode: str = "performance"


class CleanCfg(_Strict):
    visibility_min: float = 0.5
    max_gap_s: float = 0.025
    lowpass_hz: float | None = 12.0  # for event detection (velocities, ground level)
    feature_lowpass_hz: float | None = 20.0  # for feature signals (ADR-0008)
    filter_order: int = 4


class EventsCfg(_Strict):
    contact_height_frac: float = 0.03
    ground_percentile: float = 5.0
    min_stance_s: float = 0.08
    stride_time_min_s: float = 0.45
    stride_time_max_s: float = 1.0


class WindowsCfg(_Strict):
    strides: int = 20
    quality_min: float = 0.8


class RunnerCfg(_Strict):
    leg_length_cm: float | None = None


class BootstrapCfg(_Strict):
    n: int = 2000


class PipelineConfig(_Strict):
    seed: int = 0
    capture: CaptureCfg = Field(default_factory=CaptureCfg)
    pose: PoseCfg = Field(default_factory=PoseCfg)
    clean: CleanCfg = Field(default_factory=CleanCfg)
    events: EventsCfg = Field(default_factory=EventsCfg)
    windows: WindowsCfg = Field(default_factory=WindowsCfg)
    runner: RunnerCfg = Field(default_factory=RunnerCfg)
    bootstrap: BootstrapCfg = Field(default_factory=BootstrapCfg)


class TagRule(_Strict):
    feature: str
    op: Literal[">", "<"]
    threshold: float | None = None
    keywords: list[str] = Field(default_factory=list)


class FeaturesConfig(_Strict):
    feature_order: list[str]
    tags: dict[str, TagRule]

    def uncalibrated(self) -> list[str]:
        """Tag names whose threshold is still null."""
        return [name for name, rule in self.tags.items() if rule.threshold is None]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config(path: str | Path = "configs/default.yaml") -> PipelineConfig:
    return PipelineConfig.model_validate(load_yaml(path))


def load_features_config(path: str | Path = "configs/features.yaml") -> FeaturesConfig:
    return FeaturesConfig.model_validate(load_yaml(path))


def config_hash(cfg: BaseModel | dict[str, Any]) -> str:
    """Stable sha256 over the canonical JSON of a config (sorted keys)."""
    data = cfg.model_dump(mode="json") if isinstance(cfg, BaseModel) else cfg
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()
