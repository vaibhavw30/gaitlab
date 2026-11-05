"""scripts/cluster/*: budget tally, rsync filters, job headers (docs/CLUSTER.md).

Nothing here touches the network: sacct is stubbed and rsync runs between local temp dirs.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
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
    assert {f.stem for f in files} == {"rf_gate", "rf_gputest", "rf_synth"}
    limits = {"rf_gate": "00:15:00", "rf_gputest": "00:45:00", "rf_synth": "00:45:00"}
    for f in files:
        heads = set(re.findall(r"^#SBATCH (\S+)$", f.read_text(), re.M))
        assert {f"--job-name={f.stem}", "--account=bhhv-dtai-gh", "--partition=ghx4",
                "--gpus-per-node=1", "--nice=1000", f"--time={limits[f.stem]}",
                "--output=logs/%x_%j.out"} <= heads, f  # fmt: skip
        assert "source scripts/cluster/env.sh job" in f.read_text()
        # the most one GPU is billed for: 72 cores and 110 GB per GH200 (docs/CLUSTER.md)
        assert {"--cpus-per-task=72", "--mem=100g"} <= heads, f


@needs_bash
@pytest.mark.parametrize("script", sorted(p.name for p in CLUSTER.glob("*.sh")))
def test_shell_scripts_parse(script: str):
    subprocess.run(["bash", "-n", str(CLUSTER / script)], check=True)


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
    for s in sent:
        assert not re.match(r"(\.venv|\.hf_cache|runs|results|models|logs)(/|$)", s), s
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
