import numpy as np
import pytest

from gaitlab.eval import metrics as m


def test_qwk_basic_cases():
    y = [1, 2, 3, 4, 5, 3, 2]
    assert m.quadratic_weighted_kappa(y, y) == pytest.approx(1.0)
    assert m.quadratic_weighted_kappa([1, 5], [5, 1]) < 0


def test_qwk_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    a = rng.integers(1, 6, 200)
    b = np.clip(a + rng.integers(-1, 2, 200), 1, 5)
    ours = m.quadratic_weighted_kappa(a, b)
    ref = sk.cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])
    assert ours == pytest.approx(ref, abs=1e-9)


def test_agreement_helpers():
    assert m.exact_agreement([1, 2, 3], [1, 2, 4]) == pytest.approx(2 / 3)
    assert m.within_one([1, 2, 3], [2, 4, 3]) == pytest.approx(2 / 3)


def test_icc_perfect_and_noisy():
    base = np.linspace(0, 10, 20)
    assert m.icc_2_1(np.stack([base, base], axis=1)) == pytest.approx(1.0)
    noisy = np.stack([base, base + np.random.default_rng(0).normal(0, 3, 20)], axis=1)
    assert 0 < m.icc_2_1(noisy) < 1


def test_bland_altman():
    ba = m.bland_altman([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    assert ba["bias"] == pytest.approx(1.0)
    assert ba["loa_low"] == pytest.approx(1.0) and ba["loa_high"] == pytest.approx(1.0)


def test_calibration_metrics():
    y = np.array([1, 2, 3, 4, 5])
    perfect = np.eye(5)
    assert m.brier_multiclass(perfect, y) == pytest.approx(0.0)
    assert m.ece(perfect, y) == pytest.approx(0.0)
    uniform = np.full((5, 5), 0.2)
    assert m.brier_multiclass(uniform, y) == pytest.approx(0.8)


def test_macro_f1_multilabel():
    t = [{"overstride"}, {"low_cadence"}, set()]
    p = [{"overstride"}, set(), {"high_bounce"}]
    r = m.macro_f1_multilabel(t, p, ["overstride", "low_cadence", "high_bounce"])
    assert r["per_tag"]["overstride"] == pytest.approx(1.0)
    assert r["per_tag"]["low_cadence"] == pytest.approx(0.0)
    assert "high_bounce" in r["absent_in_truth"]
    assert r["macro_f1"] == pytest.approx(0.5)


def test_cluster_bootstrap_resamples_groups():
    y = np.array([1, 1, 1, 0, 0, 0])
    groups = ["a", "a", "a", "b", "b", "b"]
    ci = m.cluster_bootstrap_ci(lambda i: float(y[i].mean()), groups, n_boot=500, seed=0)
    assert ci["point"] == pytest.approx(0.5)
    assert ci["low"] <= 0.5 <= ci["high"]
    assert ci["low"] in (0.0, 0.5) and ci["high"] in (0.5, 1.0)  # only whole sessions move


def test_reliability_table_bins_by_max_probability():
    from gaitlab.eval.metrics import reliability_table

    p = np.array([[0.9, 0.1, 0, 0, 0], [0.15, 0.85, 0, 0, 0], [0.3, 0.3, 0.4, 0, 0]])
    bins = reliability_table(p, [1, 1, 3], n_bins=10)
    assert len(bins) == 10 and sum(b["count"] for b in bins) == 3
    top = bins[8]  # (0.8, 0.9]
    assert top["count"] == 2 and top["accuracy"] == 0.5
    assert top["confidence"] == pytest.approx(0.875)
    assert bins[0]["count"] == 0 and bins[0]["accuracy"] is None


def test_confusion_counts_rows_are_true_grades_columns_predicted():
    c = m.confusion_counts([1, 1, 3, 5, 5], [1, 2, 3, 4, 5])
    assert c.shape == (5, 5) and c.dtype.kind == "i"
    assert c[0].tolist() == [1, 1, 0, 0, 0]  # true 1: once right, once predicted 2
    assert c[4].tolist() == [0, 0, 0, 1, 1]
    assert c.sum() == 5 and c[1].sum() == 0  # no true 2s: empty row, still present


def test_confusion_counts_refuses_grades_outside_1_to_5():
    with pytest.raises(ValueError, match=r"1\.\.5"):
        m.confusion_counts([1, 6], [1, 2])
    with pytest.raises(ValueError, match="length"):
        m.confusion_counts([1, 2], [1])
