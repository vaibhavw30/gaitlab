from __future__ import annotations

import os
import subprocess
import sys

import pytest

from gaitlab.config import FeaturesConfig, PipelineConfig, load_features_config


@pytest.fixture
def cfg() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    return load_features_config("configs/features.yaml")


@pytest.fixture
def synthetic_features_cfg(features_cfg: FeaturesConfig) -> FeaturesConfig:
    from gaitlab.data.synthetic import synthetic_features_config

    return synthetic_features_config(features_cfg)


_CHILD = "GAITLAB_TEST_CHILD"
_TIMEOUT = "GAITLAB_CHILD_TIMEOUT_S"


def _text(b: bytes | str | None) -> str:
    return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run ``@pytest.mark.own_process`` tests in a fresh interpreter.

    torch and LightGBM load different OpenMP runtimes and segfault when both live in one
    process on macOS (ADR-0013). LightGBM tests load it into the pytest process, so tests that
    load a torch model run as ``python -m pytest <nodeid>`` in a child instead.

    With ``GAITLAB_CHILD_TIMEOUT_S`` set (the DeltaAI GPU job sets it, docs/CLUSTER.md), a child
    that runs longer is killed and the test fails. faulthandler dumps every thread's stack a
    little before the kill, so the failure shows where it hung, and the next test still runs.
    """
    if pyfuncitem.get_closest_marker("own_process") is None or os.environ.get(_CHILD):
        return None
    timeout = float(os.environ[_TIMEOUT]) if os.environ.get(_TIMEOUT) else None
    cmd = [sys.executable, "-m", "pytest", pyfuncitem.nodeid, "-q", "-rs", "-p", "no:cacheprovider"]
    if timeout:
        cmd += ["-o", f"faulthandler_timeout={max(1, int(timeout * 0.9))}"]
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, env={**os.environ, _CHILD: "1"},
            cwd=pyfuncitem.config.rootpath, timeout=timeout,
        )  # fmt: skip
    except subprocess.TimeoutExpired as exc:
        tail = (_text(exc.stdout) + _text(exc.stderr))[-6000:]
        pytest.fail(f"child pytest killed after {timeout:g} s ({_TIMEOUT}):\n{tail}", pytrace=False)
    tail = (res.stdout + res.stderr)[-6000:]
    if res.returncode != 0:
        pytest.fail(f"child pytest failed (exit {res.returncode}):\n{tail}", pytrace=False)
    if " skipped" in res.stdout and " passed" not in res.stdout:
        pytest.skip(f"skipped in child process:\n{tail[-800:]}")
    return True
