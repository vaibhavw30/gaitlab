"""models.baselines: LR + LightGBM with leave-one-session-out CV (TECHNICAL_PLAN §7.1)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from gaitlab.models import baselines as bl

requires_ml = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None or importlib.util.find_spec("lightgbm") is None,
    reason="needs the 'ml' extra (scikit-learn, LightGBM)",
)

FEATURES = ["cadence_spm", "overstride_L"]


def _toy(n_sessions: int = 5, per: int = 12, seed: int = 0):
    """Windows whose grade is a step function of overstride (grades 2..4 only; 1 and 5 absent)."""
    rng = np.random.default_rng(seed)
    rows, y, groups = [], [], []
    for s in range(n_sessions):
        for _ in range(per):
            ov = rng.uniform(0.1, 0.5)
            rows.append(
                {
                    "cadence_spm_med": 165 + rng.normal(0, 3),
                    "cadence_spm_iqr": abs(rng.normal(2, 0.5)),
                    "overstride_L_med": ov,
                    "overstride_L_iqr": np.nan if rng.random() < 0.1 else 0.02,
                }
            )
            y.append(4 if ov < 0.23 else 3 if ov < 0.37 else 2)
            groups.append(f"s{s}")
    return bl.feature_matrix(rows, FEATURES), np.array(y), np.array(groups)


@requires_ml
@pytest.mark.parametrize("kind", ["logreg", "lgbm"])
def test_predict_proba_always_has_five_grade_columns(kind: str):
    X, y, g = _toy()
    model = bl.fit_logreg(X, y) if kind == "logreg" else bl.fit_lgbm(X, y, groups=g)
    p = bl.predict_proba(model, X)
    assert p.shape == (len(X), 5)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p[:, [0, 4]] == 0).all(), "grades 1 and 5 never seen in training"
    assert set(bl.predict_grade(model, X)) <= {2, 3, 4}


@requires_ml
def test_single_grade_training_predicts_that_grade_with_certainty():
    X, _, _ = _toy(n_sessions=2, per=5)
    for fit in (bl.fit_logreg, bl.fit_lgbm):
        p = bl.predict_proba(fit(X, np.full(len(X), 3)), X)
        assert (p[:, 2] == 1).all() and p.shape == (len(X), 5)


@requires_ml
def test_loso_never_fits_on_the_held_out_session():
    X, y, g = _toy(n_sessions=4)
    seen: list[set[str]] = []

    def spy_fit(Xtr, ytr, idx):
        seen.append(set(g[idx]))
        return bl.fit_logreg(Xtr, ytr)

    oof = bl.loso_proba(spy_fit, X, y, g)
    assert oof.shape == (len(X), 5)
    assert len(seen) == 4
    for held, fitted_on in zip(sorted(set(g)), seen, strict=True):
        assert held not in fitted_on and len(fitted_on) == 3


@requires_ml
def test_lgbm_grid_is_chosen_by_loso_qwk():
    X, y, g = _toy()
    model = bl.fit_lgbm(X, y, groups=g)
    assert set(model.cv_qwk) == {bl.grid_key(p) for p in bl.LGBM_GRID}
    best = max(model.cv_qwk, key=model.cv_qwk.get)
    assert bl.grid_key(model.params) == best


@requires_ml
def test_logreg_learns_a_clean_rule_out_of_fold():
    X, y, g = _toy(per=30)
    oof = bl.loso_proba(lambda Xtr, ytr, _idx: bl.fit_logreg(Xtr, ytr), X, y, g)
    assert np.mean(oof.argmax(axis=1) + 1 == y) > 0.8  # test threshold, not a reported metric


@requires_ml
def test_tag_models_one_vs_rest():
    X, y, _ = _toy()
    tag_sets = [{"overstride"} if grade == 2 else set() for grade in y]
    vocab = ["overstride", "low_cadence"]
    tm = bl.fit_tags(X, tag_sets, vocab)
    pred = bl.predict_tags(tm, X)
    assert len(pred) == len(X)
    assert all(p <= {"overstride"} for p in pred), "a tag absent from training is never predicted"
    hits = np.mean(
        [("overstride" in p) == ("overstride" in t) for p, t in zip(pred, tag_sets, strict=True)]
    )
    assert hits > 0.8


def test_module_imports_without_loading_sklearn_or_lightgbm():
    code = (
        "import sys, gaitlab.models.baselines; "
        "assert 'sklearn' not in sys.modules and 'lightgbm' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@requires_ml
def test_baseline_predictions_fit_on_train_dev_and_score_test_only(synthetic_features_cfg):
    X, y, g = _toy(n_sessions=6, per=10)
    windows = [
        {"window_id": f"{s}:natural:{i}", "session_id": s, **dict(zip(_cols(), x, strict=True))}
        for i, (s, x) in enumerate(zip(g, X, strict=True))
    ]
    labels = {
        w["window_id"]: {"grade": int(gr), "tags": ""} for w, gr in zip(windows, y, strict=True)
    }
    split = {"s0": "train", "s1": "train", "s2": "train", "s3": "dev", "s4": "test", "s5": "test"}
    preds, meta = bl.baseline_predictions(
        windows, labels, split, synthetic_features_cfg, FEATURES, synthetic=True
    )
    assert set(preds) == {"rules", "logreg", "lgbm"}
    test_ids = {w["window_id"] for w in windows if split[w["session_id"]] == "test"}
    for name, rows in preds.items():
        assert {r["window_id"] for r in rows} == test_ids, name
        assert all(r["synthetic"] is True for r in rows)
    for r in preds["logreg"]:
        assert sum(r[f"p{k}"] for k in range(1, 6)) == pytest.approx(1.0)
    assert all(r["p1"] is None for r in preds["rules"])  # rules have no probabilities
    assert meta["train_sessions"] == ["s0", "s1", "s2", "s3"]
    assert meta["lgbm"]["params"] in bl.LGBM_GRID


def _cols() -> list[str]:
    return [f"{f}_{s}" for f in FEATURES for s in ("med", "iqr")]
