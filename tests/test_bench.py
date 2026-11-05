"""bench.trt and bench.latency (EVALUATION B1/B2, ADR-0004, TECHNICAL_PLAN §8).

The parity gate, batching and metric assembly are pure and run everywhere. Timing and TensorRT
compilation need CUDA and are ``@pytest.mark.gpu``; nothing here writes a result file.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gaitlab.bench.trt import (
    assemble_metrics,
    combine_parity,
    parity,
    read_test_prompts,
    static_batches,
)
from gaitlab.config import BenchConfig, load_bench_config
from gaitlab.llm import runtime

MODEL_270M = "google/gemma-3-270m-it"
has_torch = importlib.util.find_spec("torch") is not None
has_trt = has_torch and importlib.util.find_spec("torch_tensorrt") is not None
requires_270m = pytest.mark.skipif(
    not has_torch or not runtime.is_cached(MODEL_270M),
    reason="needs the 'llm' extra and gemma-3-270m-it in the HF cache",
)


def _onehot(grades: list[int], eps: float = 0.0) -> np.ndarray:
    p = np.full((len(grades), 5), eps)
    p[np.arange(len(grades)), np.asarray(grades) - 1] = 1.0 - 4 * eps
    return p


def _timing(p50: float) -> dict:
    return {"p50_ms": p50, "p95_ms": p50, "mean_ms": p50, "n": 3, "peak_mem_mb": 1.0}


def test_bench_yaml_loads_into_the_typed_model():
    cfg = load_bench_config("configs/bench.yaml")
    assert isinstance(cfg, BenchConfig)
    assert cfg.batch_sizes == [1, 16] and cfg.precision == "fp16"
    assert cfg.trt.ir == "dynamo" and cfg.trt.parity_required == 1.0
    with pytest.raises(ValueError):
        BenchConfig.model_validate({"precision": "int8"})


def test_static_batches_pad_the_last_chunk_by_repeating_the_last_prompt():
    chunks = static_batches(5, 2)
    assert [c.tolist() for c in chunks] == [[0, 1], [2, 3], [4, 4]]
    assert [c.tolist() for c in static_batches(2, 4)] == [[0, 1, 1, 1]]
    assert all(len(c) == 16 for c in static_batches(40, 16))
    with pytest.raises(ValueError):
        static_batches(0, 1)


def test_parity_counts_argmax_agreement_and_max_abs_dp():
    ref = _onehot([1, 2, 3, 4], eps=0.05)
    same = parity(ref, ref.copy())
    assert same == {"n_prompts": 4, "argmax_identical": 1.0, "max_abs_dp": 0.0, "n_nonfinite": 0}
    flipped = ref.copy()
    flipped[0] = _onehot([5], eps=0.05)[0]
    p = parity(ref, flipped)
    assert p["argmax_identical"] == 0.75
    assert p["max_abs_dp"] == pytest.approx(0.75)


def test_nonfinite_outputs_count_as_disagreement():
    ref = _onehot([1, 2], eps=0.05)
    bad = ref.copy()
    bad[1] = np.nan  # e.g. an FP16 overflow
    p = parity(ref, bad)
    assert p["n_nonfinite"] == 1 and p["argmax_identical"] == 0.5
    assert np.isfinite(p["max_abs_dp"])


def test_combined_parity_is_the_worst_batch_size():
    a = {"n_prompts": 4, "argmax_identical": 1.0, "max_abs_dp": 0.01, "n_nonfinite": 0}
    b = {"n_prompts": 4, "argmax_identical": 0.75, "max_abs_dp": 0.2, "n_nonfinite": 1}
    c = combine_parity({1: a, 16: b})
    assert c["argmax_identical"] == 0.75 and c["max_abs_dp"] == 0.2 and c["n_nonfinite"] == 1
    assert c["by_batch_size"] == {"1": a, "16": b}


def _assemble(argmax_identical: float) -> dict:
    par = {"n_prompts": 4, "argmax_identical": argmax_identical, "max_abs_dp": 0.01,
           "n_nonfinite": 0}  # fmt: skip
    timings = {1: {"eager": _timing(4.0), "trt": _timing(2.0)},
               16: {"eager": _timing(8.0), "trt": _timing(2.0)}}  # fmt: skip
    return assemble_metrics(
        timings, combine_parity({1: par, 16: par}), parity_required=1.0, precision="fp16",
        env={"gpu": "test"},
    )  # fmt: skip


def test_speedup_is_reported_only_when_parity_passes():
    ok = _assemble(1.0)
    assert ok["speedup_reported"] is True
    assert ok["scope"] == "grade scoring pass"
    assert ok["batch"]["1"]["speedup_p50"] == pytest.approx(2.0)
    assert ok["batch"]["16"]["speedup_mean"] == pytest.approx(4.0)
    failed = _assemble(0.75)
    assert failed["speedup_reported"] is False
    for entry in failed["batch"].values():
        assert not any(k.startswith("speedup") for k in entry)
        assert {"eager", "trt"} <= set(entry)  # timings stay, speedups go


def test_read_test_prompts_returns_user_turns_and_the_synthetic_flag(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    recs = [
        {"window_id": f"w{i}", "synthetic": True,
         "messages": [{"role": "user", "content": f"p{i}"}, {"role": "assistant", "content": "x"}]}
        for i in range(3)
    ]  # fmt: skip
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert read_test_prompts(path) == (["p0", "p1", "p2"], True)
    recs[1]["synthetic"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(ValueError, match="mix"):
        read_test_prompts(path)


def test_bench_modules_import_without_torch():
    code = (
        "import sys, gaitlab.bench.trt, gaitlab.bench.latency; "
        "bad = {'torch', 'torch_tensorrt', 'tensorrt'} & set(sys.modules); assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.own_process
@pytest.mark.skipif(not has_torch, reason="needs torch")
def test_cuda_time_refuses_without_cuda():
    import torch

    from gaitlab.bench.latency import cuda_time

    if torch.cuda.is_available():
        pytest.skip("CUDA present; covered by the gpu test")
    with pytest.raises(RuntimeError, match="CUDA"):
        cuda_time(lambda: None, warmup=1, iters=1)


# --- CUDA ------------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.own_process
def test_cuda_time_on_gpu():
    import torch

    from gaitlab.bench.latency import cuda_time

    a = torch.randn(256, 256, device="cuda")
    out = cuda_time(lambda: a @ a, warmup=2, iters=5)
    assert out["n"] == 5
    assert 0 < out["p50_ms"] <= out["p95_ms"]
    assert out["peak_mem_mb"] > 0


@pytest.mark.gpu
@pytest.mark.own_process
@requires_270m
@pytest.mark.skipif(not has_trt, reason="needs the 'trt' extra (torch-tensorrt)")
def test_bench_scoring_compiles_and_reports_parity_on_270m():
    from gaitlab.bench.trt import bench_scoring
    from gaitlab.config import load_train_lora_config
    from gaitlab.llm.grade import load_grader

    g = load_grader(MODEL_270M, cfg=load_train_lora_config(), device="cuda")
    prompts = [f"cadence_spm: {150 + 5 * i}.0 (iqr 2.0)" for i in range(6)]
    cfg = BenchConfig.model_validate({"warmup_iters": 2, "measure_iters": 3, "batch_sizes": [1, 4]})
    m = bench_scoring(g, prompts, cfg)
    assert m["precision"] == "fp16" and m["batch_sizes"] == [1, 4]
    assert set(m["batch"]) == {"1", "4"}
    assert 0.0 <= m["parity"]["argmax_identical"] <= 1.0
    assert m["parity"]["n_prompts"] == 6
    has_speedup = all("speedup_p50" in e for e in m["batch"].values())
    assert has_speedup is m["speedup_reported"]
    assert {"torch", "torch_tensorrt", "tensorrt", "gpu_name"} <= set(m["env"])
