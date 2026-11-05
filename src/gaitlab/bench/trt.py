"""Torch-TensorRT benchmark of the grade scoring pass (ADR-0004, EVALUATION B2). TASKS M4.

Contract:
``bench_scoring(grader, prompts, cfg_bench) -> dict``
1. Build ``ScoringWrapper`` (llm/grade.py) in FP16 (``cfg_bench.precision``; ``bf16`` is allowed
   because Gemma 3 can overflow in FP16), eval mode, on CUDA. This casts ``grader.model`` in place.
2. Inputs: ``grader.scoring_inputs(prompts)`` padded to ``max_prompt_tokens``; batch sizes from
   ``configs/bench.yaml`` (1 and 16). Prompts are cut into static batches; the last one is padded
   by repeating its last prompt and the padding rows are dropped (``static_batches``).
3. Compile: ``torch_tensorrt.compile(wrapper, ir="dynamo", arg_inputs=[ids, mask],
   use_explicit_typing=True, optimization_level=cfg.trt.optimization_level)``. Use explicit
   typing with an FP16 model (``enabled_precisions`` is deprecated in recent Torch-TensorRT).
   One compiled module per batch size (static shapes).
4. Time eager and compiled with ``bench.latency.cuda_time`` (warmup/iters from config) on the first
   static batch.
5. Parity over all test prompts, per batch size, against eager at the same precision: argmax
   identical fraction and max |ΔP| after softmax (``parity``); non-finite rows count as
   disagreements. The reported parity is the worst batch size (``combine_parity``). If parity <
   ``cfg.trt.parity_required`` (or any row is non-finite), ``speedup_reported = False`` and the
   speedup fields are omitted (``assemble_metrics``).
6. Return metrics for ``write_result("bench", "trt", ...)``: per batch size p50/p95/mean ms for
   both paths, speedups, parity, plus torch, torch_tensorrt, TensorRT, CUDA and cuDNN versions,
   the GPU name and its power limit (provenance also records GPU, driver and package versions).

``gaitlab bench --trt [--synthetic]`` reads the prompts from the SFT ``test.jsonl`` user turns
(``read_test_prompts``) and writes the result. torch is imported only inside ``bench_scoring``.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PRECISION_DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


def read_test_prompts(path: str | Path) -> tuple[list[str], bool]:
    """SFT JSONL → (user prompts in file order, synthetic flag). Refuses a synthetic/real mix."""
    recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not recs:
        raise ValueError(f"{path} is empty")
    flags = {bool(r.get("synthetic", False)) for r in recs}
    if len(flags) > 1:
        raise ValueError(f"{path} mixes synthetic and real records")
    prompts = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in recs]
    return prompts, flags == {True}


def static_batches(n: int, batch_size: int) -> list[np.ndarray]:
    """Index chunks of exactly ``batch_size``; the last is padded with its last index."""
    if n <= 0 or batch_size <= 0:
        raise ValueError("need at least one prompt and a positive batch size")
    out = []
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        out.append(np.concatenate([idx, np.full(batch_size - len(idx), idx[-1])]))
    return out


def parity(p_ref: np.ndarray, p_test: np.ndarray) -> dict[str, Any]:
    """Agreement of two (N, 5) probability arrays. Rows with any non-finite value disagree."""
    p_ref, p_test = np.asarray(p_ref, float), np.asarray(p_test, float)
    finite = np.isfinite(p_ref).all(axis=1) & np.isfinite(p_test).all(axis=1)
    same = finite & (p_ref.argmax(axis=1) == p_test.argmax(axis=1))
    dp = np.abs(p_ref[finite] - p_test[finite])
    return {
        "n_prompts": len(p_ref),
        "argmax_identical": float(same.mean()),
        "max_abs_dp": float(dp.max()) if dp.size else None,
        "n_nonfinite": int((~finite).sum()),
    }


def combine_parity(by_batch: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Worst case over batch sizes, plus the per-batch-size parities."""
    vals = list(by_batch.values())
    dps = [v["max_abs_dp"] for v in vals if v["max_abs_dp"] is not None]
    return {
        "n_prompts": vals[0]["n_prompts"],
        "argmax_identical": min(v["argmax_identical"] for v in vals),
        "max_abs_dp": max(dps) if dps else None,
        "n_nonfinite": max(v["n_nonfinite"] for v in vals),
        "by_batch_size": {str(bs): dict(v) for bs, v in by_batch.items()},
    }


def assemble_metrics(
    timings: Mapping[int, Mapping[str, Mapping[str, float]]],
    parity_all: Mapping[str, Any],
    *,
    parity_required: float,
    precision: str,
    env: Mapping[str, Any],
) -> dict[str, Any]:
    """The B2 metrics dict. Speedups (eager / TensorRT) appear only if the parity gate passes."""
    reported = parity_all["argmax_identical"] >= parity_required and parity_all["n_nonfinite"] == 0
    batch: dict[str, Any] = {}
    for bs in sorted(timings):
        eager, trt = dict(timings[bs]["eager"]), dict(timings[bs]["trt"])
        entry: dict[str, Any] = {"eager": eager, "trt": trt}
        if reported:
            entry["speedup_p50"] = eager["p50_ms"] / trt["p50_ms"]
            entry["speedup_mean"] = eager["mean_ms"] / trt["mean_ms"]
        batch[str(bs)] = entry
    return {
        "scope": "grade scoring pass",
        "precision": precision,
        "batch_sizes": sorted(timings),
        "batch": batch,
        "parity": dict(parity_all),
        "parity_required": parity_required,
        "speedup_reported": reported,
        "env": dict(env),
    }


def _probs(module: Callable, ids, mask, chunks: Sequence[np.ndarray], n: int) -> np.ndarray:
    import torch

    out = np.zeros((n, 5))
    for idx in chunks:
        sel = torch.as_tensor(idx, device=ids.device)
        logits = module(ids.index_select(0, sel), mask.index_select(0, sel))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        keep = ~np.r_[False, idx[1:] == idx[:-1]]  # drop the padding repeats
        out[idx[keep]] = p[keep]
    return out


def _time(module: Callable, ids, mask, cfg_bench) -> dict[str, float]:
    from gaitlab.bench.latency import cuda_time

    return cuda_time(lambda: module(ids, mask), cfg_bench.warmup_iters, cfg_bench.measure_iters)


def _env() -> dict[str, Any]:
    import tensorrt
    import torch
    import torch_tensorrt

    env: dict[str, Any] = {
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "tensorrt": tensorrt.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    try:
        env["power_limit"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()[0]  # fmt: skip
    except (subprocess.SubprocessError, FileNotFoundError, IndexError):
        env["power_limit"] = None
    return env


def bench_scoring(grader, prompts: Sequence[str], cfg_bench) -> dict[str, Any]:
    """See module docstring. ``grader`` must be on CUDA; ``cfg_bench`` is a ``BenchConfig``."""
    import torch
    import torch_tensorrt

    from gaitlab.llm import runtime

    if grader.device != "cuda":
        raise ValueError(f"bench_scoring needs a CUDA grader, got device {grader.device!r}")
    if not prompts:
        raise ValueError("no prompts to benchmark")
    dtype = PRECISION_DTYPES[cfg_bench.precision]
    wrapper = grader.scoring_wrapper().to(runtime.torch_dtype(dtype)).eval()
    grader.dtype = dtype
    enc = grader.scoring_inputs(list(prompts))
    ids, mask = enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda")
    n = len(prompts)

    timings: dict[int, dict[str, Any]] = {}
    parities: dict[int, dict[str, Any]] = {}
    for bs in cfg_bench.batch_sizes:
        chunks = static_batches(n, bs)
        first = torch.as_tensor(chunks[0], device="cuda")
        ex_ids, ex_mask = ids.index_select(0, first), mask.index_select(0, first)
        with torch.no_grad():
            compiled = torch_tensorrt.compile(
                wrapper, ir=cfg_bench.trt.ir, arg_inputs=[ex_ids, ex_mask],
                use_explicit_typing=True, optimization_level=cfg_bench.trt.optimization_level,
            )  # fmt: skip
            parities[bs] = parity(
                _probs(wrapper, ids, mask, chunks, n), _probs(compiled, ids, mask, chunks, n)
            )
            timings[bs] = {
                "eager": _time(wrapper, ex_ids, ex_mask, cfg_bench),
                "trt": _time(compiled, ex_ids, ex_mask, cfg_bench),
            }
        compiled = None  # free the engine before the next batch size
        torch.cuda.empty_cache()
    return assemble_metrics(
        timings, combine_parity(parities), parity_required=cfg_bench.trt.parity_required,
        precision=cfg_bench.precision, env=_env(),
    )  # fmt: skip
