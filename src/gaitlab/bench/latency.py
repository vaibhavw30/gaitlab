"""Stage timing (EVALUATION B1). TASKS M4.

``cpu_time(fn, warmup, iters) -> dict`` is implemented (perf_counter) for CPU stages.
``cuda_time(fn, warmup, iters) -> dict`` (implemented) uses ``torch.cuda.Event(enable_timing=True)``
around each call and synchronizes before reading; returns ms p50/p95/mean plus ``peak_mem_mb`` from
``torch.cuda.max_memory_allocated`` (reset after warmup, before measuring). Raises without CUDA.
``bench_pipeline(clip, cfg)`` (not yet written) times decode, pose, clean+events+features, scoring
pass and generate on the fixed clip and computes end-to-end fps = frames / total seconds.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    a = np.asarray(samples_ms, float)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "n": len(a),
    }


def cpu_time(fn: Callable[[], object], warmup: int = 5, iters: int = 50) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return summarize_ms(samples)


def cuda_time(fn: Callable[[], object], warmup: int = 20, iters: int = 200) -> dict[str, float]:
    """Time ``fn`` on the current CUDA device: ms per call, CUDA events, sync before each read."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("cuda_time needs a CUDA device; use cpu_time for CPU stages")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    out = summarize_ms(samples)
    out["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return out
