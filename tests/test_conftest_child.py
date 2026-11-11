"""tests/conftest.py ``own_process`` children: a hang fails one test instead of eating the job.

On DeltaAI a hung GPU test (the TensorRT engine call, CLUSTER.md) used to run until Slurm killed
the whole job. With ``GAITLAB_CHILD_TIMEOUT_S`` set, the child pytest is killed after that many
seconds, the test fails with the child's stack dump, and the remaining tests still run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HANGS = """
import time

import pytest


@pytest.mark.own_process
def test_hangs():
    time.sleep(60)


@pytest.mark.own_process
def test_after_the_hang_still_runs():
    assert True
"""


def _run(tmp_path: Path, timeout: str | None) -> subprocess.CompletedProcess:
    shutil.copy(ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_hangs.py").write_text(HANGS)
    env = dict(os.environ)
    env.pop("GAITLAB_TEST_CHILD", None)
    env.pop("GAITLAB_CHILD_TIMEOUT_S", None)
    if timeout is not None:
        env["GAITLAB_CHILD_TIMEOUT_S"] = timeout
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_hangs.py",
         "-k", "hangs or after"],
        capture_output=True, text=True, cwd=tmp_path, env=env, timeout=120,
    )  # fmt: skip


def test_a_hung_child_is_killed_with_a_stack_dump_and_the_next_test_runs(tmp_path: Path):
    res = _run(tmp_path, "4")
    out = res.stdout + res.stderr
    assert res.returncode == 1, out
    assert "1 failed, 1 passed" in out, out
    assert "killed after 4 s" in out and "GAITLAB_CHILD_TIMEOUT_S" in out, out
    # faulthandler dumped the child's stack before the kill: the hang shows where it is
    assert "most recent call first" in out and "in test_hangs" in out, out


def test_the_gpu_job_sets_a_per_test_timeout():
    gputest = (ROOT / "scripts" / "cluster" / "rf_gputest.sbatch").read_text()
    assert "export GAITLAB_CHILD_TIMEOUT_S=" in gputest
