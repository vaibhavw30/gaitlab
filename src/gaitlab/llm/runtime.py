"""Device, dtype and model-cache choices shared by ``llm.grade`` and ``llm.train_lora``.

- Device: ``cuda`` > ``mps`` > ``cpu`` unless one is asked for explicitly.
- CUDA loads bfloat16; MPS and CPU load float32. Training always uses eager attention (the
  Gemma 3 recommendation); inference uses eager on MPS/CPU and SDPA on CUDA.
- Model files live in a repo-local Hugging Face cache, ``<repo>/.hf_cache`` (git-ignored), unless
  ``HF_HOME`` is set: CLAUDE.md rule 8 keeps reads and writes inside the repository. Downloads of
  gated models (Gemma) need ``HF_TOKEN`` in the environment.

torch is imported only inside ``pick_device`` and ``torch_dtype``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

DEVICES = ("cuda", "mps", "cpu")


def select_device(*, cuda_available: bool, mps_available: bool, prefer: str | None = None) -> str:
    """Pure device choice. ``prefer`` must be one of ``DEVICES`` and available."""
    available = {"cuda": cuda_available, "mps": mps_available, "cpu": True}
    if prefer is not None:
        if prefer not in available:
            raise ValueError(f"unknown device {prefer!r}; use one of {DEVICES}")
        if not available[prefer]:
            raise ValueError(f"device {prefer!r} requested but not available")
        return prefer
    return next(d for d in DEVICES if available[d])


def pick_device(prefer: str | None = None) -> str:
    import torch

    return select_device(
        cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
        prefer=prefer,
    )


@dataclass(frozen=True)
class LoadSettings:
    dtype: str  # torch dtype name
    attn_implementation: str


def load_settings(device: str, training: bool) -> LoadSettings:
    if device == "cuda":
        return LoadSettings("bfloat16", "eager" if training else "sdpa")
    return LoadSettings("float32", "eager")


def torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def repo_root() -> Path:
    """The checkout this package runs from (editable install), else the working directory."""
    here = Path(__file__).resolve().parents[3]
    return here if (here / "pyproject.toml").exists() else Path.cwd()


def hf_home() -> Path:
    env = os.environ.get("HF_HOME")
    return Path(env) if env else repo_root() / ".hf_cache"


def hf_cache_dir() -> Path:
    return hf_home() / "hub"


def use_repo_hf_home() -> None:
    """Point every Hugging Face library default (datasets, hub) at ``hf_home()``.

    Call before the first ``transformers``/``datasets`` import; later imports read it once.
    """
    os.environ.setdefault("HF_HOME", str(hf_home()))


def is_cached(model_id: str, filename: str = "config.json") -> bool:
    """Whether ``model_id`` has ``filename`` and weights in the repo-local cache."""
    snap = hf_cache_dir() / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return any((d / filename).exists() and any(d.glob("*.safetensors")) for d in snap.glob("*"))


def from_pretrained_kwargs(model_id: str) -> dict:
    """``cache_dir`` inside the repo; no network when the files are already there."""
    return {"cache_dir": str(hf_cache_dir()), "local_files_only": is_cached(model_id)}


@contextmanager
def offline_if_cached(model_id: str) -> Iterator[None]:
    """Set ``HF_HUB_OFFLINE=1`` while ``model_id`` is fully cached (PEFT checks it at save time,
    so adapter saves don't query the Hub for a gated repo)."""
    old = os.environ.get("HF_HUB_OFFLINE")
    if is_cached(model_id):
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old
