"""scripts/cluster/*: budget tally, rsync filters, job headers (docs/CLUSTER.md).

Nothing here touches the network: sacct is stubbed and rsync runs between local temp dirs.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "scripts" / "cluster"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
needs_rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
RUN_ID = "1b-20261104T120000Z-0123abcd"


def _budget(tmp_path: Path, sacct_lines: list[str], *args: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "sacct"
    stub.write_text("#!/bin/sh\ncat <<'EOF'\n" + "\n".join(sacct_lines) + "\nEOF\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "USER": "vwudaru"}
    return subprocess.run(
        ["bash", str(CLUSTER / "budget.sh"), *args], capture_output=True, text=True, env=env
    )


def _job(jid: str, name: str, part: str, secs: int, tres: str) -> str:
    return f"{jid}|{name}|{part}|COMPLETED|{secs}|{tres}"


@needs_bash
def test_budget_counts_rf_jobs_with_gpu_memory_and_interactive_factors(tmp_path: Path):
    one = "billing=1,cpu=16,gres/gpu=1,mem=64G,node=1"
    lines = [
        _job("1", "rf_gate", "ghx4", 1800, one),  # 0.5
        _job("2", "rf_dbg", "ghx4-interactive", 1800, one),  # 0.5 h x factor 2 = 1.0
        _job("3", "rf_big", "ghx4", 1800, "cpu=16,gres/gpu=1,mem=220G,node=1"),  # 2 GH200s = 1.0
        _job("4", "other", "ghx4", 36000, one),  # not ours
        "5|rf_pend|ghx4|PENDING|0|",
    ]
    res = _budget(tmp_path, lines)
    assert res.returncode == 0, res.stderr
    assert "USED 2.50 / 15.0" in res.stdout and "REMAINING 12.50" in res.stdout
    assert "raw GPU-hours 1.50" in res.stdout
    assert "other" not in res.stdout


@needs_bash
def test_budget_stops_before_a_job_that_could_pass_fourteen_hours(tmp_path: Path):
    lines = [_job("1", "rf_synth", "ghx4", 13 * 3600 + 1800, "cpu=16,gres/gpu=1,mem=64G")]
    ok = _budget(tmp_path, lines, "--next", "00:20:00", "ghx4")
    assert ok.returncode == 0 and "tally would be 13.83" in ok.stdout
    stop = _budget(tmp_path, lines, "--next", "00:40:00", "ghx4")
    assert stop.returncode == 3 and "ASK" in stop.stdout
    interactive = _budget(tmp_path, lines, "--next", "00:20:00", "ghx4-interactive")
    assert interactive.returncode == 3, "interactive is charged double"
    past = _budget(tmp_path, [_job("1", "rf_x", "ghx4", 14 * 3600, "gres/gpu=1,cpu=1,mem=1G")])
    assert past.returncode == 3 and "ASK" in past.stdout


def test_job_files_ask_for_one_gpu_billed_as_one_gh200():
    files = sorted(CLUSTER.glob("rf_*.sbatch"))
    limits = {"rf_gate": "00:15:00", "rf_gputest": "00:45:00", "rf_synth": "00:45:00",
              "rf_trtprobe": "00:35:00", "rf_bench": "00:45:00"}  # fmt: skip
    assert {f.stem for f in files} == set(limits)
    for f in files:
        heads = set(re.findall(r"^#SBATCH (\S+)$", f.read_text(), re.M))
        assert {f"--job-name={f.stem}", "--account=bhhv-dtai-gh", "--partition=ghx4",
                "--gpus-per-node=1", "--nice=0", f"--time={limits[f.stem]}",
                "--output=logs/%x_%j.out"} <= heads, f  # fmt: skip
        assert "source scripts/cluster/env.sh job" in f.read_text()
        # the most one GPU is billed for: 72 cores and 110 GB per GH200 (docs/CLUSTER.md)
        assert {"--cpus-per-task=72", "--mem=100g"} <= heads, f


@needs_bash
@pytest.mark.parametrize("script", sorted(p.name for p in CLUSTER.glob("*.sh")))
def test_shell_scripts_parse(script: str):
    subprocess.run(["bash", "-n", str(CLUSTER / script)], check=True)


def test_scripts_run_directly_are_executable_in_git():
    """rsync -a copies the mode: a 100644 script fails on the cluster with "Permission denied"."""
    out = subprocess.run(
        ["git", "ls-files", "-s", "scripts/cluster"], capture_output=True, text=True, check=True,
        cwd=ROOT,
    ).stdout  # fmt: skip
    modes = {Path(line.split("\t")[1]).name: line.split()[0] for line in out.splitlines()}
    for name in sorted(p.name for p in CLUSTER.glob("*.sh")):
        if name == "env.sh":  # sourced, never run
            continue
        assert os.access(CLUSTER / name, os.X_OK), f"chmod +x scripts/cluster/{name}"
        assert modes.get(name, "100755") == "100755", f"git add --chmod=+x scripts/cluster/{name}"


def test_job_env_is_fully_offline():
    text = (CLUSTER / "env.sh").read_text()
    job = text.split("if [[ $GAITLAB_ENV_MODE == job ]]; then", 1)[1].split("elif", 1)[0]
    for var in ("UV_OFFLINE=1", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        assert var in job
    for var in ("UV_FROZEN=1", "UV_NO_SYNC=1", "WANDB_MODE=disabled"):
        assert f"export {var}" in text
    store = re.search(r'^GAITLAB_STORE="\$\{GAITLAB_STORE:-(.*)\}"$', text, re.M)
    assert store and store.group(1).startswith("/work/nvme/"), "store set from round 1"


@needs_bash
@needs_rsync
def test_rsync_up_dry_run_sends_code_but_no_data_caches_or_runs(tmp_path: Path):
    env = {**os.environ, "GAITLAB_RSYNC_DEST": str(tmp_path / "dest") + "/"}
    res = subprocess.run(
        ["bash", str(CLUSTER / "rsync-up.sh"), "--dry-run"], capture_output=True, text=True,
        env=env,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr
    assert "CHECK OK" in res.stdout and "nothing sent" in res.stdout
    sent = set(res.stdout.splitlines())
    assert {"src/gaitlab/cli.py", "uv.lock", "configs/train_lora.yaml"} <= sent
    assert any(s.startswith(".git/") for s in sent), ".git goes up (git_dirty provenance)"
    assert "results/.gitkeep" in sent, "tracked: without it git on the cluster reports a dirty tree"
    for s in sent:
        assert not re.match(r"(\.venv|\.hf_cache|runs|models|logs)(/|$)", s), s
        assert not s.startswith("results/") or s in ("results/", "results/.gitkeep"), s
        assert not s.startswith("data/") or s == "data/" or s.startswith("data/synthetic"), s
        assert "__pycache__" not in s and not s.endswith(".env")
    assert not (tmp_path / "dest").exists()


@needs_bash
@needs_rsync
def test_rsync_down_pulls_only_the_listed_artifacts(tmp_path: Path):
    cluster, laptop = tmp_path / "cluster", tmp_path / "laptop"
    run = cluster / "runs" / RUN_ID
    for rel in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json",
                "trainer_state.json", "run.json", "config.json", "SYNTHETIC",
                "checkpoints/checkpoint-20/model.safetensors", "big.bin"):  # fmt: skip
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text("x")
    (cluster / "logs").mkdir()
    for name in ("rf_gate_1.out", "other_2.out"):
        (cluster / "logs" / name).write_text("log")
    (cluster / "results" / "synthetic" / "bench").mkdir(parents=True)
    (cluster / "results" / "synthetic" / "bench" / "trt.json").write_text("{}")
    (cluster / "results" / "synthetic" / "notes.txt").write_text("no")
    env = {**os.environ, "GAITLAB_RSYNC_SRC": str(cluster), "GAITLAB_LOCAL_ROOT": str(laptop)}
    cmd = ["bash", str(CLUSTER / "rsync-down.sh"), RUN_ID, "--with-results"]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    got = {str(p.relative_to(laptop)) for p in laptop.rglob("*") if p.is_file()}
    kept = ("adapter/adapter_model.safetensors", "adapter/adapter_config.json",
            "trainer_state.json", "run.json", "config.json", "SYNTHETIC")  # fmt: skip
    assert got == {f"runs/{RUN_ID}/{f}" for f in kept} | {
        "logs/rf_gate_1.out",
        "results/synthetic/bench/trt.json",
    }
    bad = subprocess.run(
        ["bash", str(CLUSTER / "rsync-down.sh"), "../etc"], capture_output=True, text=True,
        env=env,
    )  # fmt: skip
    assert bad.returncode == 2


def _job_sandbox(tmp_path: Path) -> dict[str, str]:
    """$HOME/gaitlab → this repo, a fake store, and stubs for timeout/python/gaitlab/uv that log
    their argv, so a job script's shell glue runs on the laptop exactly as written."""
    home, store, bin_dir = tmp_path / "home", tmp_path / "store", tmp_path / "bin"
    (store / "venv" / "bin").mkdir(parents=True)
    bin_dir.mkdir()
    home.mkdir()
    (home / "gaitlab").symlink_to(ROOT)
    log = tmp_path / "calls.log"
    stubs = {
        bin_dir / "timeout": 'echo "timeout $*" >> "$LOG"; shift 2; exec "$@"',
        bin_dir / "uv": 'echo "uv $* CHILD_TIMEOUT=${GAITLAB_CHILD_TIMEOUT_S:-}" >> "$LOG"',
        store / "venv" / "bin" / "python": 'echo "python $*" >> "$LOG"; [ "$2" != bad ]',
        store / "venv" / "bin" / "gaitlab": 'echo "gaitlab $*" >> "$LOG"',
    }
    for path, body in stubs.items():
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    return {**os.environ, "HOME": str(home), "GAITLAB_STORE": str(store), "LOG": str(log),
            "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_JOB_ID": "1"}  # fmt: skip


def _sbatch(name: str, env: dict[str, str], *args: str) -> tuple[subprocess.CompletedProcess, str]:
    res = subprocess.run(
        ["bash", str(CLUSTER / f"{name}.sbatch"), *args], capture_output=True, text=True, env=env
    )
    log = Path(env["LOG"])
    return res, log.read_text() if log.exists() else ""


@needs_bash
def test_trtprobe_job_runs_each_case_under_its_own_timeout_on_the_venv_python(tmp_path: Path):
    env = _job_sandbox(tmp_path)
    venv_py = f"{env['GAITLAB_STORE']}/venv/bin/python"
    res, calls = _sbatch("rf_trtprobe", env)
    assert res.returncode == 0, res.stderr
    cases = ["gemma-bf16", "gemma1l-bf16"]
    assert [ln.split()[-1] for ln in calls.splitlines() if ln.startswith("python")] == cases
    # the abort signal must reach python itself: through `uv run` there'd be no stack dump
    assert f"timeout --signal=ABRT 10m {venv_py} scripts/cluster/trt_probe.py gemma-bf16" in calls
    assert all(f"PROBE {c} EXIT 0" in res.stdout for c in cases)
    res, _ = _sbatch("rf_trtprobe", _job_sandbox(tmp_path / "b"), "tiny", "bad")
    assert "PROBE tiny EXIT 0" in res.stdout and "PROBE bad EXIT 1" in res.stdout
    res, _ = _sbatch("rf_trtprobe", _job_sandbox(tmp_path / "c"), *["tiny"] * 4)
    assert res.returncode == 2 and "at most 3" in res.stderr


@needs_bash
def test_bench_and_gputest_jobs_call_what_they_say(tmp_path: Path):
    env = _job_sandbox(tmp_path)
    res, calls = _sbatch("rf_bench", env)
    assert res.returncode == 0, res.stderr
    gaitlab = f"{env['GAITLAB_STORE']}/venv/bin/gaitlab"
    assert f"timeout --signal=ABRT 40m {gaitlab} bench --trt --synthetic --model 1b" in calls
    assert "BENCH EXIT 0" in res.stdout
    res, calls = _sbatch("rf_gputest", _job_sandbox(tmp_path / "g"))
    assert res.returncode == 0, res.stderr
    assert "uv run pytest -m gpu" in calls and "CHILD_TIMEOUT=1200" in calls


def _probe_module():
    spec = importlib.util.spec_from_file_location("trt_probe", CLUSTER / "trt_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return probe


def test_trt_probe_cases_parse_before_any_gpu_import():
    probe = _probe_module()
    assert probe.parse_case("tiny") == ("tiny", "float16")
    assert probe.parse_case("gemma1l-bf16") == ("gemma1l", "bfloat16")
    for bad in ("gemma-cpp", "gemma-py-bf16", "llama", "gemma-int8", ""):
        with pytest.raises(ValueError):
            probe.parse_case(bad)
    with pytest.raises(ValueError):
        probe.main("gemma-int8")  # rejected before torch_tensorrt is imported
    cmd = probe.gdb_command("/usr/bin/gdb", 42)
    assert cmd[:5] == ["/usr/bin/gdb", "-p", "42", "-batch", "-nx"]
    assert "thread apply all bt 30" in cmd and f"echo {probe.MARK}\\n" in cmd


TARGET = """
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("trt_probe", sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
w = probe.start_watcher("gemma-bf16", float(sys.argv[2]))
if sys.argv[3] == "hang":
    time.sleep(60)  # stands in for the engine call that never returns
w.kill()
w.wait()
print("RETURNED", flush=True)
"""


def _watched(tmp_path: Path, after: str, mode: str) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "gdb": 'echo "Thread 1 (LWP 7) main: in execute_engine"; echo "==ALL THREADS=="; '
               'echo "Thread 2 worker"',
        "nvidia-smi": 'echo "utilization.gpu [%], memory.used [MiB], power.draw [W]"',
    }.items():  # fmt: skip
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    (tmp_path / "target.py").write_text(TARGET)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_JOB_ID": "9",
           "PYTHONFAULTHANDLER": "1"}  # fmt: skip
    return subprocess.run(
        [sys.executable, "target.py", str(CLUSTER / "trt_probe.py"), after, mode],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=90,
    )  # fmt: skip


def test_watcher_reports_a_hung_call_with_gpu_state_and_native_stacks_then_aborts(tmp_path):
    res = _watched(tmp_path, "1", "hang")
    assert res.returncode == -6, res.stdout + res.stderr  # SIGABRT: faulthandler's Python stack
    assert "PROBE gemma-bf16 HANG" in res.stdout
    assert "PROBE gemma-bf16 GPU utilization.gpu" in res.stdout
    assert "PROBE gemma-bf16 GDB Thread 1 (LWP 7) main: in execute_engine" in res.stdout
    assert "Thread 2 worker" not in res.stdout, "all threads go to the file, not the log"
    saved = tmp_path / "logs" / "rf_trtprobe_9_gemma-bf16_gdb.txt"
    assert "Thread 2 worker" in saved.read_text()
    assert "Fatal Python error: Aborted" in res.stderr and "time.sleep" not in res.stdout


def test_watcher_stays_silent_when_the_call_returns(tmp_path: Path):
    res = _watched(tmp_path, "5", "fast")
    assert res.returncode == 0 and "RETURNED" in res.stdout, res.stderr
    assert "HANG" not in res.stdout and not (tmp_path / "logs").exists()


@pytest.mark.own_process
@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs the 'llm' extra (torch)"
)
def test_every_probe_case_builds_and_runs_eagerly_on_cpu():
    """What a probe job builds before TensorRT ever runs, so a broken case fails here, not after
    a queue wait. Needs gemma-3-270m-it in the HF cache (skips otherwise)."""
    import torch

    from gaitlab.llm import runtime

    if not runtime.is_cached("google/gemma-3-270m-it"):
        pytest.skip("needs gemma-3-270m-it in the HF cache")
    probe = _probe_module()
    assert set(probe.BUILDERS) == set(probe.MODELS)
    for name in probe.MODELS:
        model, dtype = probe.parse_case(f"{name}-bf16")
        module, args, extra = probe.BUILDERS[model](dtype, device="cpu")
        with torch.no_grad():
            out = module(*args)
        assert bool(torch.isfinite(out).all()), name
        if name == "gemmatorchops":
            assert torch.ops.aten.embedding.default in extra["torch_executed_ops"]
