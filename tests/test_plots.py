"""viz.plots: ladder, knee-by-condition, confusion, reliability and Bland-Altman charts, read
from result JSON, strides parquet and the V7 stride-pairs parquet."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaitlab.report.results import write_result
from gaitlab.viz.plots import (
    SERIES_LIGHT,
    bland_altman_rows,
    confusion_rows,
    knee_bands,
    ladder_rows,
    reliability_rows,
    series_color,
)

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


# --- confusion matrix (G1 per model) -----------------------------------------------------------


def _confusion(diag: int) -> dict:
    counts = np.zeros((5, 5), dtype=int)
    np.fill_diagonal(counts, diag)
    counts[2, 3] = 2  # two true 3s predicted 4
    counts[1] = 0  # no true 2s in test
    return {"grades": [1, 2, 3, 4, 5], "counts": counts.tolist()}


@pytest.fixture
def ladder_with_confusion(tmp_path: Path) -> Path:
    models = {m: {"qwk": _ci(q), "confusion": _confusion(d)}
              for m, q, d in (("rules", 0.9, 4), ("logreg", 0.7, 3))}  # fmt: skip
    metrics = {"split": "test", "order": ["rules", "logreg"], "models": models,
               "best": {"model": "rules", "qwk": 0.9}}  # fmt: skip
    return write_result("grader", "ladder", metrics, config={}, data_manifest="sha256:x",
                        synthetic=True, root=tmp_path / "results")  # fmt: skip


def test_confusion_rows_default_to_the_best_model(ladder_with_confusion: Path):
    payload = json.loads(ladder_with_confusion.read_text())
    counts, model, prov = confusion_rows(payload)
    assert model == "rules" and prov.synthetic is True
    assert counts.shape == (5, 5) and counts[2, 3] == 2 and counts[0, 0] == 4
    counts, model, _ = confusion_rows(payload, "logreg")
    assert model == "logreg" and counts[0, 0] == 3


def test_confusion_rows_refuse_unknown_models_and_old_ladder_files(ladder_with_confusion: Path):
    payload = json.loads(ladder_with_confusion.read_text())
    with pytest.raises(ValueError, match="lgbm"):
        confusion_rows(payload, "lgbm")
    for block in payload["metrics"]["models"].values():  # as written before confusion existed
        del block["confusion"]
    with pytest.raises(ValueError, match="gaitlab eval"):
        confusion_rows(payload)


@requires_mpl
def test_confusion_chart_writes_png_and_svg(ladder_with_confusion: Path, tmp_path: Path):
    from gaitlab.viz.plots import confusion_chart

    paths = confusion_chart(ladder_with_confusion, tmp_path / "plots" / "confusion")
    assert sorted(p.suffix for p in paths) == [".png", ".svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    with pytest.raises(ValueError, match="synthetic"):
        confusion_chart(ladder_with_confusion, tmp_path / "docs" / "assets" / "confusion")


# --- reliability diagram (G5) ------------------------------------------------------------------


def _bins(filled: dict[int, tuple[int, float, float]]) -> list[dict]:
    """Ten equal-width bins; ``filled`` maps bin index → (count, confidence, accuracy)."""
    out = []
    for i in range(10):
        n, c, a = filled.get(i, (0, None, None))
        out.append({"low": i / 10, "high": (i + 1) / 10, "count": n, "confidence": c,
                    "accuracy": a})  # fmt: skip
    return out


@pytest.fixture
def calibration_json(tmp_path: Path) -> Path:
    models = {
        "lgbm": {"ece": 0.1, "brier": 0.2, "reliability": _bins({9: (6, 0.95, 1.0)})},
        "logreg": {"ece": 0.05, "brier": 0.1,
                   "reliability": _bins({5: (2, 0.55, 0.5), 8: (4, 0.85, 0.75)})},
    }  # fmt: skip
    metrics = {"split": "test", "models": models, "no_probabilities": ["rules"], "ece_bins": 10}
    return write_result("grader", "calibration", metrics, config={}, data_manifest="sha256:x",
                        synthetic=True, root=tmp_path / "results")  # fmt: skip


def test_reliability_rows_skip_empty_bins_in_ladder_order(calibration_json: Path):
    rows, prov = reliability_rows(json.loads(calibration_json.read_text()))
    assert prov.synthetic is True
    assert list(rows) == ["logreg", "lgbm"]  # ladder order, not file order
    conf, acc, count = rows["logreg"]
    assert conf.tolist() == [0.55, 0.85] and acc.tolist() == [0.5, 0.75]
    assert count.tolist() == [2, 4]
    only, _ = reliability_rows(json.loads(calibration_json.read_text()), ["lgbm"])
    assert list(only) == ["lgbm"]


def test_reliability_rows_refuse_models_without_probabilities(calibration_json: Path):
    payload = json.loads(calibration_json.read_text())
    with pytest.raises(ValueError, match="rules"):
        reliability_rows(payload, ["rules"])
    payload["metrics"]["models"] = {}
    with pytest.raises(ValueError, match="probabilities"):
        reliability_rows(payload)


def test_series_color_follows_the_model_not_its_rank():
    assert series_color("lgbm") == SERIES_LIGHT[2]  # third rung of the ladder
    assert series_color("rules") == SERIES_LIGHT[0]
    assert series_color("gemma3_270m_lora") == SERIES_LIGHT[5]
    assert series_color("zz_extra") in SERIES_LIGHT[6:]
    assert len(set(SERIES_LIGHT)) == len(SERIES_LIGHT) == 8


@requires_mpl
def test_reliability_chart_writes_png_and_svg(calibration_json: Path, tmp_path: Path):
    from gaitlab.viz.plots import reliability_chart

    paths = reliability_chart(calibration_json, tmp_path / "plots" / "reliability")
    assert sorted(p.suffix for p in paths) == [".png", ".svg"]
    assert all(p.stat().st_size > 0 for p in paths)
    with pytest.raises(ValueError, match="synthetic"):
        reliability_chart(calibration_json, tmp_path / "docs" / "reliability")


@requires_mpl
def test_real_results_from_a_dirty_tree_never_go_to_docs(calibration_json: Path, tmp_path: Path):
    """A chart under docs/ is a report: it needs synthetic false and git_dirty false (rule 1)."""
    from gaitlab.viz.plots import reliability_chart

    payload = json.loads(calibration_json.read_text())
    payload.update(synthetic=False, git_dirty=True)
    dirty = tmp_path / "dirty.json"  # a test input, not a results/ file
    dirty.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dirty"):
        reliability_chart(dirty, tmp_path / "docs" / "assets" / "reliability")
    assert reliability_chart(dirty, tmp_path / "plots" / "reliability")  # outside docs/: fine


# --- Bland-Altman (V7 backend agreement) -------------------------------------------------------


@pytest.fixture
def agreement(tmp_path: Path) -> tuple[Path, Path]:
    """A V7 result and its pairs parquet (cadence: a = b + 2; knee: a = b)."""
    import hashlib

    n = 6
    pairs = pd.DataFrame({
        "session_id": ["s1"] * n, "clip_id": ["natural"] * n, "side": ["l"] * n,
        "t_ic_a": np.arange(n) * 0.7, "t_ic_b": np.arange(n) * 0.7,
        "cadence_spm_a": 172.0 + np.arange(n), "cadence_spm_b": 170.0 + np.arange(n),
        "knee_flex_ic_deg_a": [10.0, 12, np.nan, 14, 15, 16],
        "knee_flex_ic_deg_b": [10.0, 12, 13, 14, 15, 16], "synthetic": [True] * n,
    })  # fmt: skip
    path = tmp_path / "backend_pairs.mediapipe-rtmpose.parquet"
    pairs.to_parquet(path, index=False)
    metrics = {
        "backends": ["mediapipe", "rtmpose"], "n_pairs": n,
        "features": {"cadence_spm": {"bias": 2.0, "loa_low": 2.0, "loa_high": 2.0, "n": n},
                     "knee_flex_ic_deg": {"bias": 0.0, "loa_low": 0.0, "loa_high": 0.0, "n": 5}},
        "pairs_file": path.name, "pairs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }  # fmt: skip
    res = write_result(
        "validation", "backend_agreement", metrics, config={}, data_manifest="sha256:x",
        synthetic=True, root=tmp_path / "results",
    )  # fmt: skip
    return res, path


def test_bland_altman_rows_pair_means_and_differences(agreement):
    res, pairs = agreement
    rows, prov = bland_altman_rows(json.loads(res.read_text()), pd.read_parquet(pairs))
    assert prov.synthetic is True and list(rows) == ["cadence_spm", "knee_flex_ic_deg"]
    mean, diff, summary = rows["cadence_spm"]
    assert mean.tolist() == [171.0, 172.0, 173.0, 174.0, 175.0, 176.0]
    assert diff.tolist() == [2.0] * 6 and summary["bias"] == 2.0
    mean, diff, _ = rows["knee_flex_ic_deg"]
    assert len(mean) == 5  # the NaN pair is left out
    only, _ = bland_altman_rows(json.loads(res.read_text()), pd.read_parquet(pairs),
                                ["knee_flex_ic_deg"])  # fmt: skip
    assert list(only) == ["knee_flex_ic_deg"]
    with pytest.raises(ValueError, match="trunk_lean_deg"):
        bland_altman_rows(json.loads(res.read_text()), pd.read_parquet(pairs), ["trunk_lean_deg"])


@requires_mpl
def test_bland_altman_chart_writes_png_and_svg_and_checks_the_pairs_file(agreement, tmp_path):
    from gaitlab.viz.plots import bland_altman_chart

    res, pairs = agreement
    paths = bland_altman_chart(res, pairs, tmp_path / "plots" / "agreement")
    assert sorted(p.suffix for p in paths) == [".png", ".svg"]
    with pytest.raises(ValueError, match="synthetic"):
        bland_altman_chart(res, pairs, tmp_path / "docs" / "agreement")
    pd.read_parquet(pairs).head(3).to_parquet(pairs, index=False)  # changed after the result
    with pytest.raises(ValueError, match="sha256"):
        bland_altman_chart(res, pairs, tmp_path / "plots" / "agreement")
