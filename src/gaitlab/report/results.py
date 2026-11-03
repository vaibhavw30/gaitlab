"""The only writer of result files (ADR-0007, TECHNICAL_PLAN §9)."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gaitlab.config import config_hash

SCHEMA_VERSION = 1
TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "mediapipe",
    "rtmlib",
    "torch",
    "transformers",
    "peft",
    "trl",
    "tensorrt",
    "torch-tensorrt",
    "scikit-learn",
    "lightgbm",
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def git_state() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else True}


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return out


def hardware() -> dict[str, str | None]:
    info: dict[str, str | None] = {"cpu": platform.processor() or platform.machine(), "gpu": None}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        if q:
            name, driver = (s.strip() for s in q.splitlines()[0].split(","))
            info.update(gpu=name, driver=driver)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    try:
        import torch  # noqa: PLC0415

        info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def manifest_hash(paths: list[str | Path]) -> str:
    """sha256 over (relative path, file sha256) pairs, sorted. Used as data_manifest_hash."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        fh = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        h.update(f"{p}:{fh}\n".encode())
    return "sha256:" + h.hexdigest()


def write_result(
    area: str,
    name: str,
    metrics: dict[str, Any],
    *,
    config: BaseModel | dict[str, Any],
    data_manifest: str,
    synthetic: bool,
    notes: str = "",
    root: str | Path = "results",
) -> Path:
    """Write ``results/<area>/<name>.json`` (or ``results/synthetic/<area>/...``) with provenance.

    Parameters
    ----------
    area : "validation" | "grader" | "bench" | ...
    name : file stem.
    metrics : JSON-serializable metrics.
    config : the run's config (hashed, not stored in full).
    data_manifest : output of ``manifest_hash`` for the inputs.
    synthetic : True for anything derived from synthetic data or rule labels (never reported).
    """
    base = Path(root) / ("synthetic" if synthetic else "") / area
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "area": area,
        "name": name,
        "synthetic": synthetic,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        **git_state(),
        "config_hash": config_hash(config),
        "data_manifest_hash": data_manifest,
        "hardware": hardware(),
        "versions": versions(),
        "notes": notes,
        "metrics": metrics,
    }
    path = base / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=float) + "\n")
    return path


def load_reportable(root: str | Path = "results") -> list[dict[str, Any]]:
    """All result files eligible for the README: not synthetic, not dirty."""
    out = []
    for p in sorted(Path(root).glob("*/*.json")):
        if p.parts[-2] == "synthetic":
            continue
        d = json.loads(p.read_text())
        if d.get("synthetic") or d.get("git_dirty"):
            continue
        d["_path"] = str(p)
        out.append(d)
    return out
