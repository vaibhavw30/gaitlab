"""viz.plots: ladder chart and knee-by-condition, read from result JSON and strides parquet."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaitlab.report.results import write_result
from gaitlab.viz.plots import knee_bands, ladder_rows

requires_mpl = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="needs the 'viz' extra (matplotlib)"
)


def _ci(point: float) -> dict[str, float]:
    return {"point": point, "low": point - 0.1, "high": point + 0.05, "n_boot": 50}


@pytest.fixture
def ladder_json(tmp_path: Path) -> Path:
    metrics = {
        "split": "test",
        "order": ["rules", "logreg", "lgbm"],
        "models": {m: {"qwk": _ci(q)} for m, q in (("lgbm", 0.6), ("rules", 0.9), ("logreg", 0.7))},
        "best": {"model": "rules", "qwk": 0.9},
    }
    return write_result(
        "grader", "ladder", metrics, config={}, data_manifest="sha256:x", synthetic=True,
        root=tmp_path / "results",
    )  # fmt: skip


def test_ladder_rows_follow_ladder_order(ladder_json: Path):
    import json

    rows, synthetic = ladder_rows(json.loads(ladder_json.read_text()))
    assert synthetic is True
    assert [r[0] for r in rows] == ["rules", "logreg", "lgbm"]
    assert rows[1][1:] == pytest.approx((0.7, 0.6, 0.75))


@requires_mpl
def test_ladder_chart_writes_png_and_svg(ladder_json: Path, tmp_path: Path):
    from gaitlab.viz.plots import ladder_chart

    paths = ladder_chart(ladder_json, tmp_path / "plots" / "ladder")
    assert sorted(p.suffix for p in paths) == [".png", ".svg"]
    assert all(p.stat().st_size > 0 for p in paths)


@requires_mpl
def test_synthetic_charts_never_go_to_docs(ladder_json: Path, tmp_path: Path):
    from gaitlab.viz.plots import ladder_chart

    with pytest.raises(ValueError, match="synthetic"):
        ladder_chart(ladder_json, tmp_path / "docs" / "assets" / "ladder")


def _strides(tmp_path: Path) -> Path:
    x = np.linspace(0, 1, 101)
    rows = []
    for cond, amp in (("natural", 40.0), ("overstride", 30.0)):
        for k in range(5):
            rows.append(
                {"session_id": "20260101-01", "clip_id": cond, "condition": cond, "stride_idx": k,
                 "knee_flex_cycle_deg": amp * np.sin(np.pi * x) + k, "synthetic": True}
            )  # fmt: skip
    path = tmp_path / "strides.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_knee_bands_median_and_iqr_per_condition(tmp_path: Path):
    bands, synthetic = knee_bands(pd.read_parquet(_strides(tmp_path)))
    assert synthetic is True
    assert set(bands) == {"natural", "overstride"}
    med, q25, q75 = bands["natural"]
    assert med.shape == (101,)
    assert med[50] == pytest.approx(42.0)  # amplitude 40 + median offset 2
    assert q25[50] == pytest.approx(41.0) and q75[50] == pytest.approx(43.0)


@requires_mpl
def test_knee_by_condition_writes_png_and_svg(tmp_path: Path):
    from gaitlab.viz.plots import knee_by_condition

    paths = knee_by_condition([_strides(tmp_path)], tmp_path / "plots" / "knee")
    assert sorted(p.suffix for p in paths) == [".png", ".svg"]
