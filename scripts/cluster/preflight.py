"""LOGIN NODE, right before sbatch: catch what would otherwise fail after a queue wait.

    $GAITLAB_STORE/venv/bin/python scripts/cluster/preflight.py probe|gputest|synth|bench

CPU only, a few seconds: no GPU, no network (HF_HUB_OFFLINE is forced), no model weights loaded
(only tokenizers). Prints one ``PREFLIGHT <check> OK|WARN|FAIL <detail>`` line per check and
exits 1 if any check FAILs. Each FAIL names the fix. docs/CLUSTER.md "Before every sbatch".

What it can't see: anything that needs the GPU itself (the TensorRT engine hang, CUDA errors).
"""

from __future__ import annotations

import importlib.metadata as md
import os
import resource
import socket
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path

JOBS = ("probe", "gputest", "synth", "bench")
SFT = Path("data/synthetic/sft")

Result = tuple[str, str]  # (OK | WARN | FAIL, detail)


def requirements(job: str) -> dict[str, list[str]]:
    """Model keys, packages and files the job reads. Model keys index train_lora.yaml ``models``."""
    if job not in JOBS:
        raise ValueError(f"job must be one of {JOBS}, got {job!r}")
    base = ["torch", "transformers"]
    trt = ["tensorrt", "torch-tensorrt"]
    return {
        "probe": {"models": ["270m"], "packages": base + trt, "files": []},
        "gputest": {
            "models": ["270m"],
            "packages": [*base, *trt, "peft", "trl", "pytest"],
            "files": [],
        },
        "synth": {
            "models": ["1b"],
            "packages": [*base, "peft", "trl"],
            "files": [str(SFT / f"{s}.jsonl") for s in ("train", "dev", "test")],
        },
        "bench": {
            "models": ["1b"],
            "packages": [*base, *trt, "peft"],
            "files": [str(SFT / "test.jsonl")],
        },
    }[job]


def check_host(hostname: str) -> Result:
    if hostname.startswith("gh-login"):
        return "OK", hostname
    if hostname.startswith("dt-login"):
        return "FAIL", f"{hostname} is Delta, not DeltaAI: ssh dtai-login.delta.ncsa.illinois.edu"
    return "FAIL", f"{hostname} is not a DeltaAI login node (gh-login*)"


def check_git_clean(porcelain: str | None, *, writes_result: bool) -> Result:
    if porcelain is None:
        return "FAIL", "git status failed: is ~/gaitlab/.git there? (rsync sends it)"
    if not porcelain.strip():
        return "OK", "tracked files match HEAD"
    files = " ".join(line[3:] for line in porcelain.splitlines())
    fix = "git checkout -- results/.gitkeep" if "results/.gitkeep" in files else "rsync again"
    # a result file written from a dirty tree records git_dirty=true and can never be reported
    return ("FAIL" if writes_result else "WARN"), f"dirty: {files} (fix: {fix})"


def check_python(version: tuple[int, int], include_dir: str) -> Result:
    if version != (3, 11):
        return "FAIL", f"python {version[0]}.{version[1]}, want 3.11 (UV_PYTHON=3.11)"
    if not (Path(include_dir) / "Python.h").exists():
        return "FAIL", (
            f"no Python.h in {include_dir}: Triton can't compile its helper. Rebuild the venv "
            "on uv's own CPython (UV_PYTHON_PREFERENCE=only-managed, docs/CLUSTER.md)"
        )
    return "OK", f"3.11 with Python.h ({include_dir})"


def check_env(env: dict[str, str], repo: Path) -> Result:
    store = env.get("GAITLAB_STORE")
    if not store:
        return "FAIL", "GAITLAB_STORE unset: run the exports from docs/CLUSTER.md first"
    hf = env.get("HF_HOME")
    if not hf or Path(hf).resolve() == (repo / ".hf_cache").resolve():
        return "FAIL", "HF_HOME unset: jobs would look for models in ~/gaitlab/.hf_cache (empty)"
    missing = [v for v in ("UV_PROJECT_ENVIRONMENT", "TRITON_CACHE_DIR") if not env.get(v)]
    if missing:
        return "FAIL", f"unset: {' '.join(missing)} (run the exports from docs/CLUSTER.md)"
    if env.get("GAITLAB_TEST_CHILD"):
        return (
            "FAIL",
            "GAITLAB_TEST_CHILD is set: sbatch would pass it on and turn off the "
            "per-test timeouts (unset GAITLAB_TEST_CHILD)",
        )
    return "OK", f"store={store} HF_HOME={hf}"


def check_venv(prefix: str, env: dict[str, str]) -> Result:
    want = env.get("UV_PROJECT_ENVIRONMENT")
    if want and Path(prefix).resolve() != Path(want).resolve():
        return "FAIL", f"running {prefix}, jobs use {want}: run this with $UV_PROJECT_ENVIRONMENT"
    return "OK", prefix


def check_packages(names: list[str]) -> Result:
    found, missing = [], []
    for n in names:
        try:
            found.append(f"{n} {md.version(n)}")
        except md.PackageNotFoundError:
            missing.append(n)
    if missing:
        return "FAIL", f"missing {' '.join(missing)} (uv sync --extra ml --extra trt ...)"
    return "OK", ", ".join(found)


def check_precision(precision: str, uses_bench: bool) -> Result:
    if uses_bench and precision == "fp16":
        return "FAIL", "configs/bench.yaml precision fp16: Gemma 3 overflows, all NaN (ADR-0015)"
    return "OK", precision


def check_core_limit(soft: int) -> Result:
    if soft == 0:
        return "OK", "core files off"
    return "WARN", "core files on (jobs inherit it); the rf_*.sbatch files set ulimit -c 0"


def check_files(paths: list[str]) -> Result:
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        return "FAIL", f"missing {' '.join(missing)} (rsync data/synthetic from the laptop)"
    return "OK", f"{len(paths)} file(s)" if paths else "none needed"


def check_logs_dir(path: Path) -> Result:
    if path.is_dir() and os.access(path, os.W_OK):
        return "OK", str(path)
    return (
        "FAIL",
        f"{path} missing: Slurm can't open the job's output and the job dies with no "
        "log (mkdir -p ~/gaitlab/logs)",
    )


def check_synthetic_run(runs_root: Path, model_key: str) -> Result:
    from gaitlab.llm.train_lora import latest_run

    try:
        run = latest_run(runs_root, model_key, synthetic=True)
    except FileNotFoundError as e:
        return "FAIL", f"{e} (sbatch scripts/cluster/rf_synth.sbatch first)"
    adapter = run / "adapter"
    need = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        return "FAIL", f"newest synthetic run {run.name} has no {' '.join(missing)}"
    return "OK", f"newest synthetic {model_key} run {run.name}"


def check_model(model_id: str, grade_tokens: list[str]) -> Result:
    """Offline: weights in HF_HOME, tokenizer loads, grade tokens score as single tokens."""
    from gaitlab.llm import runtime
    from gaitlab.llm.grade import grade_tokens_for, load_tokenizer

    if not runtime.is_cached(model_id):
        return "FAIL", f"{model_id} not in {runtime.hf_cache_dir()} (hf download {model_id})"
    try:
        tok = load_tokenizer(model_id)
        gt = grade_tokens_for(tok, grade_tokens)
    except Exception as e:
        return "FAIL", f"{model_id}: {type(e).__name__}: {e}"
    return "OK", f"{model_id} cached offline, grade tokens {gt.form}"


def _git_porcelain() -> str | None:
    try:
        return subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
            text=True, check=True, timeout=30,
        ).stdout  # fmt: skip
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _torch_cuda_build() -> Result:
    import torch

    if torch.version.cuda is None:
        return "FAIL", f"torch {torch.__version__} is a CPU build: the venv has the wrong wheel"
    return "OK", f"torch {torch.__version__} built for CUDA {torch.version.cuda}"


def run(job: str) -> list[tuple[str, str, str]]:
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"  # as in the job
    from gaitlab.config import load_bench_config, load_train_lora_config

    req = requirements(job)
    repo = Path.cwd()
    env = dict(os.environ)
    checks: list[tuple[str, Callable[[], Result]]] = [
        ("host", lambda: check_host(socket.gethostname())),
        (
            "repo",
            lambda: (
                ("OK", str(repo))
                if (repo / "pyproject.toml").is_file()
                else ("FAIL", f"run from ~/gaitlab, not {repo}")
            ),
        ),
        ("git", lambda: check_git_clean(_git_porcelain(), writes_result=job == "bench")),
        ("logs_dir", lambda: check_logs_dir(repo / "logs")),
        ("env", lambda: check_env(env, repo)),
        ("venv", lambda: check_venv(sys.prefix, env)),
        ("python", lambda: check_python(sys.version_info[:2], sysconfig.get_paths()["include"])),
        ("packages", lambda: check_packages(req["packages"])),
        ("torch_cuda_build", _torch_cuda_build),
        ("core_limit", lambda: check_core_limit(resource.getrlimit(resource.RLIMIT_CORE)[0])),
        (
            "configs",
            lambda: (
                ("OK", "train_lora.yaml, bench.yaml validate")
                if (load_train_lora_config(), load_bench_config())
                else ("FAIL", "")
            ),
        ),
        (
            "bench_precision",
            lambda: check_precision(load_bench_config().precision, job in ("bench", "gputest")),
        ),
        ("files", lambda: check_files(req["files"])),
    ]
    if job == "bench":
        checks.append(("synthetic_run", lambda: check_synthetic_run(repo / "runs", "1b")))
    lcfg = None
    for key in req["models"]:

        def model_check(key: str = key) -> Result:
            nonlocal lcfg
            lcfg = lcfg or load_train_lora_config()
            return check_model(lcfg.models[key], list(lcfg.grade.grade_tokens))

        checks.append((f"model_{key}", model_check))
    out = []
    for name, fn in checks:
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        out.append((name, status, detail))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1 or argv[0] not in JOBS:
        print(f"usage: preflight.py {{{'|'.join(JOBS)}}}", file=sys.stderr)
        return 2
    results = run(argv[0])
    for name, status, detail in results:
        print(f"PREFLIGHT {name} {status} {detail}", flush=True)
    failed = [n for n, s, _ in results if s == "FAIL"]
    print(f"PREFLIGHT RESULT {'FAIL ' + ' '.join(failed) if failed else 'PASS'} ({argv[0]})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
