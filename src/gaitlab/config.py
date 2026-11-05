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


class LoraCfg(_Strict):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]
    )  # fmt: skip


class TrainCfg(_Strict):
    epochs: int = 3
    lr: float = 2.0e-4
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    per_device_batch_size: int = 4
    grad_accum_steps: int = 4
    bf16: bool = True  # CUDA only; MPS and CPU train in float32 (llm.runtime)
    assistant_only_loss: bool = True
    seed: int = 0
    max_seq_len: int = 384


class GradeCfg(_Strict):
    max_prompt_tokens: int = 256
    grade_tokens: list[str] = Field(default_factory=lambda: [" 1", " 2", " 3", " 4", " 5"])


class GenerateCfg(_Strict):
    max_new_tokens: int = 64
    do_sample: bool = False


class ConsistencyCfg(_Strict):
    """G6: grade scoring repeated ``runs`` times; tags and cue sampled ``runs`` times at T."""

    runs: int = 5
    temperature: float = 0.7


class LoggingCfg(_Strict):
    wandb_project: str | None = "gaitlab"


class TrainLoraConfig(_Strict):
    models: dict[str, str] = Field(
        default_factory=lambda: {"1b": "google/gemma-3-1b-it", "270m": "google/gemma-3-270m-it"}
    )
    lora: LoraCfg = Field(default_factory=LoraCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    grade: GradeCfg = Field(default_factory=GradeCfg)
    generate: GenerateCfg = Field(default_factory=GenerateCfg)
    consistency: ConsistencyCfg = Field(default_factory=ConsistencyCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)


class TrtCfg(_Strict):
    ir: Literal["dynamo"] = "dynamo"
    optimization_level: int = 3
    parity_required: float = 1.0  # fraction of test prompts with identical argmax grade


class BenchConfig(_Strict):
    """configs/bench.yaml (EVALUATION §4, ADR-0004)."""

    warmup_iters: int = 20
    measure_iters: int = 200
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 16])
    precision: Literal["fp16", "bf16"] = "fp16"
    clip: str = "data/raw/BENCH/natural.mp4"
    stages: list[str] = Field(
        default_factory=lambda: [
            "decode", "pose", "clean_events_features", "scoring_pass", "generate",
        ]
    )  # fmt: skip
    trt: TrtCfg = Field(default_factory=TrtCfg)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config(path: str | Path = "configs/default.yaml") -> PipelineConfig:
    return PipelineConfig.model_validate(load_yaml(path))


def load_features_config(path: str | Path = "configs/features.yaml") -> FeaturesConfig:
    return FeaturesConfig.model_validate(load_yaml(path))


def load_train_lora_config(path: str | Path = "configs/train_lora.yaml") -> TrainLoraConfig:
    return TrainLoraConfig.model_validate(load_yaml(path))


def load_bench_config(path: str | Path = "configs/bench.yaml") -> BenchConfig:
    return BenchConfig.model_validate(load_yaml(path))


def config_hash(cfg: BaseModel | dict[str, Any]) -> str:
    """Stable sha256 over the canonical JSON of a config (sorted keys)."""
    data = cfg.model_dump(mode="json") if isinstance(cfg, BaseModel) else cfg
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()
