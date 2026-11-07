"""CLUSTER (GPU job): does a Torch-TensorRT engine actually execute on this node? (docs/CLUSTER.md)

    python scripts/cluster/trt_probe.py tiny-cpp | tiny-py | gemma-cpp | gemma-py

``tiny`` is a 3-layer MLP; ``gemma`` is the Gemma 3 270M grade-scoring wrapper exactly as
``bench.trt.bench_scoring`` builds it (FP16, explicit typing). ``cpp`` is Torch-TensorRT's default
C++ runtime (``torch.ops.tensorrt.execute_engine``), ``py`` its Python runtime
(``use_python_runtime=True``). Run each case under ``timeout``: a hang shows as a missing
``RUN OK`` line. Prints only pass/fail and module types; no timings, no result files.
"""

from __future__ import annotations

import sys


def _tiny():
    import torch

    m = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 5))
    m = m.to("cuda", torch.float16).eval()
    return m, [torch.randn(4, 64, device="cuda", dtype=torch.float16)], {"min_block_size": 1}


def _gemma():
    from gaitlab.config import load_train_lora_config
    from gaitlab.llm import runtime
    from gaitlab.llm.grade import load_grader

    g = load_grader("google/gemma-3-270m-it", cfg=load_train_lora_config(), device="cuda")
    w = g.scoring_wrapper().to(runtime.torch_dtype("float16")).eval()
    g.dtype = "float16"
    enc = g.scoring_inputs(["cadence_spm: 160.0 (iqr 2.0)"])
    return w, [enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda")], {}


def main(case: str) -> int:
    import torch
    import torch_tensorrt

    model, runtime_kind = case.split("-")
    module, args, extra = {"tiny": _tiny, "gemma": _gemma}[model]()
    say = lambda msg: print(f"PROBE {case} {msg}", flush=True)  # noqa: E731
    say(f"START torch_tensorrt {torch_tensorrt.__version__}")
    with torch.no_grad():
        ref = module(*args)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        torch.cuda.synchronize()
        say("EAGER OK")
        trt = torch_tensorrt.compile(
            module, ir="dynamo", arg_inputs=args, use_explicit_typing=True,
            use_python_runtime=(runtime_kind == "py"), **extra,
        )  # fmt: skip
        say("COMPILED " + ",".join(sorted({type(m).__name__ for m in trt.modules()})))
        out = trt(*args)
        out = out[0] if isinstance(out, (tuple, list)) else out
        torch.cuda.synchronize()
        same = bool((out.float().argmax(-1) == ref.float().argmax(-1)).all())
        say(f"RUN OK finite={bool(torch.isfinite(out).all())} argmax_same={same}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
