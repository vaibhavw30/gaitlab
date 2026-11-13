"""CLUSTER (GPU job): does a Torch-TensorRT engine run on this node, and if not, where is it stuck?

    python scripts/cluster/trt_probe.py <model>[-<precision>]    e.g. tiny gemma-bf16 gemma1l-bf16

Models:
- ``tiny``: a 3-layer MLP (float input; ran fine on the GH200, job 3550685).
- ``gemma``: the Gemma 3 270M grade-scoring wrapper exactly as ``bench.trt.bench_scoring``
  builds it (int64 ids + mask in, 5 logits out).
- Bisection of a ``gemma`` hang, one suspect each:
  ``gemma1l`` only the first decoder layer; ``emb`` only the token-embedding lookup (int64 ids,
  the 262k-row table); ``gemmanomask`` ids only, no attention mask or cumsum position ids;
  ``gemmatorchops`` the full wrapper with embedding, cumsum, index_select and index left to PyTorch
  (``torch_executed_ops``), so the engine holds only the dense layers.
- Jobs 3581071 / 3581191: ``gemma`` and ``gemma1l`` hang; ``emb``, ``gemmanomask`` and
  ``gemmatorchops`` run. In the no-mask graph every cumsum and index op depends on shapes only
  (folded at compile time); in the mask wrapper one cumsum (position ids) and two index ops
  (padding-mask lookup) depend on the runtime ``attention_mask``. One suspect each:
  ``gemmacumsumtorch`` leaves only ``aten.cumsum`` to PyTorch; ``gemmaindextorch`` only
  ``aten.index.Tensor``.
Precision ``fp16`` (default) or ``bf16``; Gemma 3 overflows fp16, so its fp16 outputs are NaN by
design (ADR-0015).

torch-tensorrt 2.14 has one merged runtime (``use_python_runtime`` is deprecated and ignored), so
there is no runtime to choose. Each engine's ``requires_output_allocator`` flag and bindings are
printed after compiling. The first engine call runs under a watcher *process* (a Python thread may
never run while the call holds the GIL): if the call hasn't returned after ``TRT_PROBE_HANG_S``
seconds (default 120), the watcher prints GPU utilisation and the native stacks (gdb), saves all
threads to ``logs/``, then sends SIGABRT so faulthandler adds the Python stack.

Lines: ``PROBE <case> START / EAGER OK / COMPILED / ENGINE / RUN OK`` or ``HANG / GPU / GDB``.
Prints pass/fail and diagnostics only; no timings go into any doc, no result files.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

DTYPES = {"fp16": "float16", "bf16": "bfloat16"}
MODELS = (
    "tiny", "gemma", "gemma1l", "emb", "gemmanomask", "gemmatorchops", "gemmacumsumtorch",
    "gemmaindextorch",
)  # fmt: skip
MARK = "==ALL THREADS=="


def parse_case(case: str) -> tuple[str, str]:
    """``gemma1l-bf16`` → ("gemma1l", "bfloat16"). Raises ValueError on anything else."""
    parts = case.split("-")
    if len(parts) == 1:
        parts.append("fp16")
    if len(parts) != 2 or parts[0] not in MODELS:
        raise ValueError(f"case must be ({'|'.join(MODELS)})[-(fp16|bf16)], got {case!r}")
    if parts[1] not in DTYPES:
        raise ValueError(f"precision must be fp16 or bf16, got {parts[1]!r}")
    return parts[0], DTYPES[parts[1]]


def _say(case: str, msg: str) -> None:
    print(f"PROBE {case} {msg}", flush=True)


# --- the watcher (a separate process) ---------------------------------------------------------


def gdb_command(gdb: str, pid: int) -> list[str]:
    """Main thread in full, every thread's current frame, then all threads (saved to a file)."""
    ex = [
        "set pagination off", "set print thread-events off", "set debuginfod enabled off",
        "info threads", "thread 1", "bt 60", f"echo {MARK}\\n", "thread apply all bt 30",
    ]  # fmt: skip
    return [gdb, "-p", str(pid), "-batch", "-nx", *[a for e in ex for a in ("-ex", e)]]


def watch(pid: int, after_s: float, case: str, out_dir: Path) -> int:
    """Wait for the engine call in process ``pid`` (our parent); on a hang, report and abort it."""
    deadline = time.monotonic() + after_s
    while time.monotonic() < deadline:
        if os.getppid() != pid:  # the parent finished or died: nothing to report
            return 0
        time.sleep(0.5)
    _say(case, f"HANG first engine call still running after {after_s:g} s; diagnostics follow")
    smi = shutil.which("nvidia-smi")
    if smi:
        q = "--query-gpu=utilization.gpu,memory.used,power.draw"
        r = subprocess.run([smi, q, "--format=csv"], capture_output=True, text=True, timeout=30)
        for line in r.stdout.strip().splitlines():
            _say(case, f"GPU {line}")
    gdb = shutil.which("gdb")
    if gdb is None:
        _say(case, "GDB not found; Python stack only")
    else:
        env = {**os.environ, "DEBUGINFOD_URLS": ""}
        try:
            r = subprocess.run(
                gdb_command(gdb, pid), capture_output=True, text=True, timeout=300, env=env
            )
            text = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            text = "gdb timed out"
        out_dir.mkdir(parents=True, exist_ok=True)
        full = out_dir / f"rf_trtprobe_{os.environ.get('SLURM_JOB_ID', 'local')}_{case}_gdb.txt"
        full.write_text(text)
        for line in text.split(MARK)[0].splitlines():
            if line.strip():
                _say(case, f"GDB {line}")
        _say(case, f"GDB all threads saved to {full}")
    try:
        os.kill(pid, signal.SIGABRT)  # faulthandler (PYTHONFAULTHANDLER=1) prints the Python stack
    except ProcessLookupError:
        pass
    return 0


def allow_ptrace_from_children() -> None:
    """Yama ``ptrace_scope=1`` lets a process be traced only by its ancestors; the watcher is a
    child. PR_SET_PTRACER_ANY lifts that for this process only (no-op without Yama or off Linux)."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(0x59616D61, ctypes.c_ulong(2**64 - 1), 0, 0, 0)  # PR_SET_PTRACER, ANY
    except OSError:
        pass


def start_watcher(case: str, after_s: float) -> subprocess.Popen:
    allow_ptrace_from_children()
    cmd = [sys.executable, __file__, "--watch", str(os.getpid()), str(after_s), case]
    return subprocess.Popen(cmd)


# --- the probe ---------------------------------------------------------------------------------


def _tiny(dtype: str, device: str = "cuda"):
    import torch

    dt = getattr(torch, dtype)
    m = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 5))
    m = m.to(device, dt).eval()
    return m, [torch.randn(4, 64, device=device, dtype=dt)], {"min_block_size": 1}


def _grader(dtype: str, device: str, layers: int | None = None):
    """The 270M grader, cast to ``dtype``, optionally cut to its first ``layers`` layers."""
    from gaitlab.config import load_train_lora_config
    from gaitlab.llm import runtime
    from gaitlab.llm.grade import load_grader

    g = load_grader("google/gemma-3-270m-it", cfg=load_train_lora_config(), device=device)
    if layers is not None:
        cfg = g.model.config
        g.model.model.layers = g.model.model.layers[:layers]
        cfg.num_hidden_layers = layers
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = cfg.layer_types[:layers]
    g.model.to(runtime.torch_dtype(dtype))
    g.dtype = dtype
    enc = g.scoring_inputs(["cadence_spm: 160.0 (iqr 2.0)"])
    return g, enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _bench_extra() -> dict:
    from gaitlab.config import load_bench_config

    return {"optimization_level": load_bench_config().trt.optimization_level}  # as the bench


def _gemma(dtype: str, device: str = "cuda", layers: int | None = None):
    g, ids, mask = _grader(dtype, device, layers)
    return g.scoring_wrapper().eval(), [ids, mask], _bench_extra()


def _emb(dtype: str, device: str = "cuda"):
    import torch

    g, ids, _ = _grader(dtype, device)

    class Embed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = g.model.model.embed_tokens  # scaled lookup, as in the model

        def forward(self, input_ids):
            return self.embed(input_ids)

    return Embed().eval(), [ids], {**_bench_extra(), "min_block_size": 1}


def _gemma_nomask(dtype: str, device: str = "cuda"):
    import torch

    g, ids, _ = _grader(dtype, device)

    class NoMask(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = g.model
            self.register_buffer("grade_ids", torch.tensor(list(g.tokens.ids)), persistent=False)

        def forward(self, input_ids):
            out = self.model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
            return out.logits[:, -1, :].index_select(-1, self.grade_ids)

    return NoMask().to(device).eval(), [ids], _bench_extra()


def _gemma_torchops(dtype: str, device: str = "cuda"):
    import torch

    module, args, extra = _gemma(dtype, device)
    aten = torch.ops.aten
    # every gather-like op in the exported graph (checked with torch.export on CPU)
    ops = {
        aten.embedding.default,
        aten.cumsum.default,
        aten.index_select.default,
        aten.index.Tensor,
    }
    return module, args, {**extra, "torch_executed_ops": ops}


def _gemma_one_op_in_torch(op_name: str):
    def build(dtype: str, device: str = "cuda"):
        import torch

        module, args, extra = _gemma(dtype, device)
        name, overload = op_name.split(".")
        op = getattr(getattr(torch.ops.aten, name), overload)
        return module, args, {**extra, "torch_executed_ops": {op}}

    return build


BUILDERS = {
    "tiny": _tiny,
    "gemma": _gemma,
    "gemma1l": lambda d, device="cuda": _gemma(d, device, layers=1),
    "emb": _emb,
    "gemmanomask": _gemma_nomask,
    "gemmatorchops": _gemma_torchops,
    "gemmacumsumtorch": _gemma_one_op_in_torch("cumsum.default"),
    "gemmaindextorch": _gemma_one_op_in_torch("index.Tensor"),
}


def engine_lines(trt) -> list[str]:
    """One line per TensorRT engine submodule: class, output-allocator flag, bindings."""
    out = []
    for name, m in trt.named_modules():
        if "TensorRTModule" in type(m).__name__:
            out.append(
                f"ENGINE {name or '.'} {type(m).__name__} "
                f"requires_output_allocator={getattr(m, 'requires_output_allocator', '?')} "
                f"inputs={getattr(m, 'input_binding_names', '?')} "
                f"outputs={getattr(m, 'output_binding_names', '?')}"
            )
    return out


def main(case: str) -> int:
    model, dtype = parse_case(case)  # before the slow imports: a typo fails fast
    import torch
    import torch_tensorrt
    from torch_tensorrt._features import ENABLED_FEATURES

    module, args, extra = BUILDERS[model](dtype)
    say = lambda msg: _say(case, msg)  # noqa: E731
    say(f"START torch_tensorrt {torch_tensorrt.__version__} features {ENABLED_FEATURES}")
    with torch.no_grad():
        ref = module(*args)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        torch.cuda.synchronize()
        say(f"EAGER OK finite={bool(torch.isfinite(ref).all())}")
        trt = torch_tensorrt.compile(
            module, ir="dynamo", arg_inputs=args, use_explicit_typing=True, **extra
        )
        say("COMPILED " + ",".join(sorted({type(m).__name__ for m in trt.modules()})))
        for line in engine_lines(trt):
            say(line)
        watcher = start_watcher(case, float(os.environ.get("TRT_PROBE_HANG_S", "120")))
        try:
            out = trt(*args)
            out = out[0] if isinstance(out, (tuple, list)) else out
            torch.cuda.synchronize()
        finally:
            watcher.kill()
            watcher.wait()
        same = bool((out.float().argmax(-1) == ref.float().argmax(-1)).all())
        say(f"RUN OK finite={bool(torch.isfinite(out).all())} argmax_same={same}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--watch":
        sys.exit(watch(int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], Path("logs")))
    sys.exit(main(sys.argv[1]))
