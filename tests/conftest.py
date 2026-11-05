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


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run ``@pytest.mark.own_process`` tests in a fresh interpreter.

    torch and LightGBM load different OpenMP runtimes and segfault when both live in one
    process on macOS (ADR-0013). LightGBM tests load it into the pytest process, so tests that
    load a torch model run as ``python -m pytest <nodeid>`` in a child instead.
    """
    if pyfuncitem.get_closest_marker("own_process") is None or os.environ.get(_CHILD):
        return None
    res = subprocess.run(
        [sys.executable, "-m", "pytest", pyfuncitem.nodeid, "-q", "-rs", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        env={**os.environ, _CHILD: "1"},
        cwd=pyfuncitem.config.rootpath,
    )
    tail = (res.stdout + res.stderr)[-6000:]
    if res.returncode != 0:
        pytest.fail(f"child pytest failed (exit {res.returncode}):\n{tail}", pytrace=False)
    if " skipped" in res.stdout and " passed" not in res.stdout:
        pytest.skip(f"skipped in child process:\n{tail[-800:]}")
    return True
