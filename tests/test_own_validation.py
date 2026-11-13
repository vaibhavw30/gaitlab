"""eval.own_validation: V5 direction checks, V6 stability ICC, V7 backend agreement (TASKS M2).

Unit tests use hand-made window / stride tables with known answers. The integration tests run
the real pipeline on ``gaitlab synth`` keypoints (whose conditions differ from natural in known
directions) and check that results land only under ``results/synthetic/validation/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaitlab.eval.own_validation import (
    EXPECTED_DIRECTIONS,
    backend_agreement,
    direction_check,
    match_strides,
    session_differences,
    stability,
    stability_matrix,
)

CFG = "configs/default.yaml"


def _windows(rows: list[tuple[str, str, int, float]], feature: str = "cadence_spm"):
    """(session, condition, start_stride, value) → a windows table with ``<feature>_med``."""
    return pd.DataFrame(
        [
            {"session_id": s, "clip_id": c, "condition": c, "start_stride": k, f"{feature}_med": v}
            for s, c, k, v in rows
        ]
    )


# --- V5 directions ----------------------------------------------------------------------------


def test_expected_directions_are_the_evaluation_table():
    assert EXPECTED_DIRECTIONS == {
        "metronome": {"cadence_spm": 1, "overstride_L": -1, "vert_osc_L": -1},
        "overstride": {"overstride_L": 1, "shin_angle_deg": 1},
    }


def test_session_differences_are_condition_median_minus_natural_median():
    w = _windows([
        ("s1", "natural", 0, 160.0), ("s1", "natural", 20, 170.0), ("s1", "natural", 40, 200.0),
        ("s1", "metronome", 0, 180.0), ("s1", "metronome", 20, 184.0),
        ("s2", "natural", 0, 165.0), ("s2", "metronome", 0, 175.0),
        ("s3", "natural", 0, 150.0),  # no metronome clip: left out
    ])  # fmt: skip
    diffs = session_differences(w, "metronome", "cadence_spm")
    assert diffs == {"s1": pytest.approx(12.0), "s2": pytest.approx(10.0)}


def test_direction_check_reports_sign_size_and_session_bootstrap_ci():
    w = _windows([
        ("s1", "natural", 0, 160.0), ("s1", "metronome", 0, 172.0),
        ("s2", "natural", 0, 165.0), ("s2", "metronome", 0, 175.0),
        ("s3", "natural", 0, 170.0), ("s3", "metronome", 0, 178.0),
    ])  # fmt: skip
    r = direction_check(w, "metronome", "cadence_spm", expected=1, n_boot=200, seed=0)
    assert r["expected_sign"] == 1 and r["n_sessions"] == 3
    assert r["diff"]["point"] == pytest.approx(10.0)  # mean of 12, 10, 8
    assert 8.0 <= r["diff"]["low"] <= r["diff"]["point"] <= r["diff"]["high"] <= 12.0
    assert r["sign_ok"] is True and r["ci_on_expected_side"] is True
    assert r["n_windows"] == {"condition": 3, "natural": 3}
    r = direction_check(w, "metronome", "cadence_spm", expected=-1, n_boot=200, seed=0)
    assert r["sign_ok"] is False and r["ci_on_expected_side"] is False


def test_direction_check_without_paired_sessions_says_so():
    w = _windows([("s1", "natural", 0, 160.0)])
    r = direction_check(w, "metronome", "cadence_spm", expected=1, n_boot=50, seed=0)
    assert r["n_sessions"] == 0 and r["diff"] is None and r["sign_ok"] is None


# --- V6 stability -----------------------------------------------------------------------------


def test_stability_matrix_rows_are_session_condition_cells_in_time_order():
    w = _windows([
        ("s1", "natural", 40, 3.0), ("s1", "natural", 0, 1.0), ("s1", "natural", 20, 2.0),
        ("s1", "metronome", 0, 5.0), ("s1", "metronome", 20, 6.0),
        ("s2", "natural", 0, 7.0), ("s2", "natural", 20, np.nan),  # NaN → cell dropped
        ("s2", "metronome", 0, 9.0),  # one window → cell dropped
    ])  # fmt: skip
    ratings, cells, dropped = stability_matrix(w, "cadence_spm")
    assert cells == [("s1", "metronome"), ("s1", "natural")]
    assert ratings.tolist() == [[5.0, 6.0], [1.0, 2.0]]  # k = smallest cell; earliest windows
    assert sorted(dropped) == [("s2", "metronome"), ("s2", "natural")]


def test_stability_icc_is_one_when_windows_repeat_exactly_and_lower_with_noise():
    rng = np.random.default_rng(0)
    rows, noisy = [], []
    for s in ("s1", "s2", "s3"):
        for c, level in (("natural", 160.0), ("metronome", 180.0)):
            base = level + rng.normal(0, 5)
            for k in range(3):
                rows.append((s, c, 20 * k, base))
                noisy.append((s, c, 20 * k, base + rng.normal(0, 5)))
    exact = stability(_windows(rows), ["cadence_spm"])["cadence_spm"]
    assert exact["icc"] == pytest.approx(1.0) and exact["n_cells"] == 6 and exact["k"] == 3
    assert stability(_windows(noisy), ["cadence_spm"])["cadence_spm"]["icc"] < 1.0


def test_stability_with_fewer_than_two_cells_is_not_computed():
    w = _windows([("s1", "natural", 0, 1.0), ("s1", "natural", 20, 2.0)])
    r = stability(w, ["cadence_spm"])["cadence_spm"]
    assert r["icc"] is None and "2 cells" in r["reason"]


# --- V7 backend agreement ---------------------------------------------------------------------


def _strides(t_ic: list[float], value: float = 0.0, side: str = "l", clip: str = "natural"):
    return pd.DataFrame(
        [
            {
                "session_id": "s1",
                "clip_id": clip,
                "side": side,
                "t_ic": t,
                "stride_time_s": 0.7,
                "cadence_spm": 170.0 + value + i,
            }
            for i, t in enumerate(t_ic)
        ]
    )


def test_strides_match_one_to_one_by_nearest_initial_contact():
    a = _strides([1.00, 1.70, 2.40, 3.10])
    b = _strides([1.02, 1.69, 2.90, 3.12])  # 2.40 has no partner within half a stride (0.35 s)
    pairs = match_strides(a, b)
    assert sorted(pairs) == [(0, 0), (1, 1), (3, 3)]


def test_strides_on_different_sides_or_clips_never_match():
    a = _strides([1.0, 1.7])
    assert match_strides(a, _strides([1.0, 1.7], side="r")) == []
    assert match_strides(a, _strides([1.0, 1.7], clip="metronome")) == []


def test_backend_agreement_is_bland_altman_on_matched_strides():
    a = _strides([1.0, 1.7, 2.4, 3.1], value=2.0)
    b = _strides([1.0, 1.7, 2.4, 9.9])  # the last stride has no partner
    r, pairs = backend_agreement(a, b, ["cadence_spm"])
    assert r["n_pairs"] == 3 and r["unmatched"] == {"a": 1, "b": 1}
    ba = r["features"]["cadence_spm"]
    assert ba["bias"] == pytest.approx(2.0) and ba["n"] == 3
    assert ba["loa_low"] == pytest.approx(2.0) and ba["loa_high"] == pytest.approx(2.0)
    assert pairs["cadence_spm_a"].tolist() == [172.0, 173.0, 174.0]  # per pair, for the chart
    assert pairs["cadence_spm_b"].tolist() == [170.0, 171.0, 172.0]
    assert pairs["t_ic_a"].tolist() == pairs["t_ic_b"].tolist() == [1.0, 1.7, 2.4]


def test_a_feature_with_no_finite_pairs_has_no_bias():
    a = _strides([1.0, 1.7])
    a["cadence_spm"] = np.nan
    r, _ = backend_agreement(a, _strides([1.0, 1.7]), ["cadence_spm"])
    assert r["features"]["cadence_spm"] == {"bias": None, "loa_low": None, "loa_high": None,
                                            "n": 0}  # fmt: skip


# --- integration on gaitlab synth keypoints ---------------------------------------------------


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    """Two synthetic sessions as npz + sessions.csv, plus a jittered copy as a second backend."""
    from gaitlab.data.synthetic import save_synthetic_keypoints, synthetic_session_clips
    from gaitlab.types import KeypointSeq

    root = tmp_path_factory.mktemp("own")
    clips = synthetic_session_clips(2, seed=3)
    save_synthetic_keypoints(clips, root)
    rng = np.random.default_rng(1)
    for npz in sorted((root / "interim").glob("*/*.synthetic.npz")):
        seq = KeypointSeq.load(str(npz))
        seq.xy = seq.xy + rng.normal(0, 1.0, seq.xy.shape).astype(seq.xy.dtype)
        seq.save(str(npz).replace(".synthetic.npz", ".jitter.npz"))
    return root


def test_run_on_synthetic_sessions_finds_the_expected_directions(synth_root: Path):
    from gaitlab.eval.own_validation import run

    out = run(
        str(synth_root / "sessions.csv"), backend="synthetic", other="jitter", synthetic=True,
        config_path=CFG, n_boot=100, results_root=str(synth_root / "res"),
    )  # fmt: skip
    assert set(out) == {"directions", "stability", "backend_agreement", "pairs"}
    pairs = out.pop("pairs")
    assert pairs == synth_root / "processed" / "backend_pairs.synthetic-jitter.parquet"
    for name, path in out.items():
        assert path == synth_root / "res" / "synthetic" / "validation" / f"{name}.json"
        assert json.loads(path.read_text())["synthetic"] is True
    d = json.loads(out["directions"].read_text())["metrics"]
    assert d["backend"] == "synthetic" and d["n_sessions"] == 2
    for cond, feats in EXPECTED_DIRECTIONS.items():
        for f in feats:
            assert d["conditions"][cond][f]["sign_ok"] is True, (cond, f)
    s = json.loads(out["stability"].read_text())["metrics"]
    assert s["features"]["cadence_spm"]["n_cells"] >= 2
    a = json.loads(out["backend_agreement"].read_text())["metrics"]
    assert a["backends"] == ["synthetic", "jitter"] and a["n_pairs"] > 0
    assert set(a["features"]) == set(s["features"])
    assert a["pairs_file"] == pairs.name and len(pd.read_parquet(pairs)) == a["n_pairs"]
    assert pd.read_parquet(pairs)["synthetic"].all()


def test_run_without_the_synthetic_flag_refuses_synthetic_keypoints(synth_root: Path):
    from gaitlab.eval.own_validation import run

    with pytest.raises(ValueError, match="synthetic"):
        run(str(synth_root / "sessions.csv"), backend="synthetic", synthetic=False,
            config_path=CFG, n_boot=10, results_root=str(synth_root / "res2"))  # fmt: skip


def test_run_without_a_second_backend_skips_agreement(synth_root: Path):
    from gaitlab.eval.own_validation import run

    out = run(str(synth_root / "sessions.csv"), backend="synthetic", synthetic=True,
              config_path=CFG, n_boot=10, results_root=str(synth_root / "res3"))  # fmt: skip
    assert set(out) == {"directions", "stability"}


def test_run_names_a_missing_npz(synth_root: Path):
    from gaitlab.eval.own_validation import run

    with pytest.raises(FileNotFoundError, match="gaitlab extract"):
        run(str(synth_root / "sessions.csv"), backend="rtmpose", synthetic=True,
            config_path=CFG, n_boot=10, results_root=str(synth_root / "res4"))  # fmt: skip


def test_validate_own_cli(synth_root: Path):
    from typer.testing import CliRunner

    from gaitlab.cli import app

    res = CliRunner().invoke(app, [
        "validate", "own", "--sessions", str(synth_root / "sessions.csv"), "--backend",
        "synthetic", "--other", "jitter", "--synthetic", "--n-boot", "10", "--results",
        str(synth_root / "res5"), "--config", CFG,
    ])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert (synth_root / "res5" / "synthetic" / "validation" / "directions.json").exists()
    res = CliRunner().invoke(app, ["validate", "biocv"])  # still waits on the license
    assert res.exit_code == 2
