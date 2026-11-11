"""scripts/cluster/preflight.py: the login-node check before every sbatch (docs/CLUSTER.md)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts/cluster/preflight.py")
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def test_every_job_names_what_it_reads():
    assert pf.requirements("bench")["models"] == ["1b"]
    assert "torch-tensorrt" in pf.requirements("gputest")["packages"]
    assert pf.requirements("synth")["files"] == [
        "data/synthetic/sft/train.jsonl", "data/synthetic/sft/dev.jsonl",
        "data/synthetic/sft/test.jsonl",
    ]  # fmt: skip
    with pytest.raises(ValueError):
        pf.requirements("train")


def test_host_must_be_a_deltaai_login_node_not_delta():
    assert pf.check_host("gh-login01")[0] == "OK"
    status, detail = pf.check_host("dt-login02")
    assert status == "FAIL" and "Delta, not DeltaAI" in detail
    assert pf.check_host("my-mac")[0] == "FAIL"


def test_dirty_tree_fails_only_jobs_that_write_a_result_and_names_the_fix():
    assert pf.check_git_clean("", writes_result=True)[0] == "OK"
    dirty = " D results/.gitkeep\n"
    assert pf.check_git_clean(dirty, writes_result=False)[0] == "WARN"
    status, detail = pf.check_git_clean(dirty, writes_result=True)
    assert status == "FAIL" and "git checkout -- results/.gitkeep" in detail
    assert pf.check_git_clean(None, writes_result=False)[0] == "FAIL"


def test_python_needs_311_and_its_headers(tmp_path: Path):
    assert pf.check_python((3, 12), str(tmp_path))[0] == "FAIL"
    status, detail = pf.check_python((3, 11), str(tmp_path))
    assert status == "FAIL" and "Python.h" in detail and "only-managed" in detail
    (tmp_path / "Python.h").write_text("")
    assert pf.check_python((3, 11), str(tmp_path))[0] == "OK"


def test_env_needs_the_store_exports_and_no_child_bypass(tmp_path: Path):
    good = {"GAITLAB_STORE": "/s", "HF_HOME": "/s/hf", "UV_PROJECT_ENVIRONMENT": "/s/venv",
            "TRITON_CACHE_DIR": "/s/t"}  # fmt: skip
    assert pf.check_env(good, tmp_path)[0] == "OK"
    assert pf.check_env({**good, "HF_HOME": ""}, tmp_path)[0] == "FAIL"
    assert pf.check_env({**good, "HF_HOME": str(tmp_path / ".hf_cache")}, tmp_path)[0] == "FAIL"
    assert pf.check_env({k: v for k, v in good.items() if k != "TRITON_CACHE_DIR"},
                        tmp_path)[0] == "FAIL"  # fmt: skip
    status, detail = pf.check_env({**good, "GAITLAB_TEST_CHILD": "1"}, tmp_path)
    assert status == "FAIL" and "timeouts" in detail


def test_venv_must_be_the_one_jobs_use(tmp_path: Path):
    assert pf.check_venv(str(tmp_path), {"UV_PROJECT_ENVIRONMENT": str(tmp_path)})[0] == "OK"
    assert pf.check_venv("/usr", {"UV_PROJECT_ENVIRONMENT": str(tmp_path)})[0] == "FAIL"


def test_fp16_bench_is_refused_and_core_files_warn():
    assert pf.check_precision("fp16", uses_bench=True)[0] == "FAIL"
    assert pf.check_precision("fp16", uses_bench=False)[0] == "OK"
    assert pf.check_precision("bf16", uses_bench=True)[0] == "OK"
    assert pf.check_core_limit(0)[0] == "OK" and pf.check_core_limit(-1)[0] == "WARN"


def test_files_logs_and_packages(tmp_path: Path):
    assert pf.check_files([])[0] == "OK"
    assert pf.check_files([str(tmp_path / "nope.jsonl")])[0] == "FAIL"
    assert pf.check_logs_dir(tmp_path / "logs")[0] == "FAIL"
    (tmp_path / "logs").mkdir()
    assert pf.check_logs_dir(tmp_path / "logs")[0] == "OK"
    assert pf.check_packages(["numpy"])[0] == "OK"
    assert pf.check_packages(["numpy", "no-such-package-x"])[0] == "FAIL"


def test_bench_needs_a_synthetic_run_with_an_adapter(tmp_path: Path):
    assert pf.check_synthetic_run(tmp_path, "1b")[0] == "FAIL"
    run = tmp_path / "1b-20261106T161038Z-b9c09dee"
    (run / "adapter").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"synthetic": True}))
    status, detail = pf.check_synthetic_run(tmp_path, "1b")
    assert status == "FAIL" and "adapter_model.safetensors" in detail
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        (run / "adapter" / f).write_text("x")
    assert pf.check_synthetic_run(tmp_path, "1b") == ("OK", f"newest synthetic 1b run {run.name}")


def test_on_the_laptop_it_runs_every_check_and_fails_on_the_host():
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HOME", "GAITLAB_STORE")}
    res = subprocess.run(
        [sys.executable, "scripts/cluster/preflight.py", "probe"], capture_output=True,
        text=True, cwd=ROOT, env=env, timeout=300,
    )  # fmt: skip
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("PREFLIGHT ")]
    names = [ln.split()[1] for ln in lines]
    assert names[:3] == ["host", "repo", "git"] and names[-1] == "RESULT", res.stdout + res.stderr
    assert "model_270m" in names and "Traceback" not in res.stderr, res.stderr
    assert res.returncode == 1 and "PREFLIGHT host FAIL" in res.stdout
    assert "PREFLIGHT env FAIL GAITLAB_STORE unset" in res.stdout
    usage = subprocess.run([sys.executable, "scripts/cluster/preflight.py"], cwd=ROOT,
                           capture_output=True, text=True)  # fmt: skip
    assert usage.returncode == 2
