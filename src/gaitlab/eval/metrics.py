"""Evaluation metrics (EVALUATION.md). Pure NumPy; no sklearn dependency at import time."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np

GRADES = np.arange(1, 6)


def exact_agreement(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(yt == yp))


def within_one(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(yt - yp) <= 1))


def quadratic_weighted_kappa(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] = tuple(GRADES)
) -> float:
    """Cohen's kappa with quadratic weights over a fixed label set (default 1..5)."""
    labels = list(labels)
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    o = np.zeros((k, k))
    for a, b in zip(y_true, y_pred, strict=True):
        o[idx[int(a)], idx[int(b)]] += 1
    n = o.sum()
    if n == 0:
        return float("nan")
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / n
    denom = (w * e).sum()
    if denom == 0:
        return 1.0 if (w * o).sum() == 0 else 0.0
    return float(1.0 - (w * o).sum() / denom)


def mae(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.nanmean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def bland_altman(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Bias (mean of a − b) and 95% limits of agreement."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    diff = diff[np.isfinite(diff)]
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    return {"bias": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd, "n": len(diff)}


def icc_2_1(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    ``ratings`` is (n_targets, k_raters_or_repeats) with no NaNs.
    """
    x = np.asarray(ratings, float)
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def macro_f1_multilabel(
    true_sets: Sequence[set[str]], pred_sets: Sequence[set[str]], vocab: Sequence[str]
) -> dict[str, object]:
    """Per-tag F1 and macro-F1 over tags that appear at least once in ``true_sets``."""
    per: dict[str, float] = {}
    for tag in vocab:
        tp = sum(1 for t, p in zip(true_sets, pred_sets, strict=True) if tag in t and tag in p)
        fp = sum(1 for t, p in zip(true_sets, pred_sets, strict=True) if tag not in t and tag in p)
        fn = sum(1 for t, p in zip(true_sets, pred_sets, strict=True) if tag in t and tag not in p)
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        per[tag] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    absent = [t for t in vocab if t not in per]
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    return {"macro_f1": macro, "per_tag": per, "absent_in_truth": absent}


def brier_multiclass(probs: np.ndarray, y_true: Sequence[int]) -> float:
    """Mean over samples of the sum of squared errors vs one-hot (grades 1..5 → columns 0..4)."""
    p = np.asarray(probs, float)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(p)), np.asarray(y_true) - 1] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def ece(probs: np.ndarray, y_true: Sequence[int], n_bins: int = 10) -> float:
    """Expected calibration error on max-probability confidence, equal-width bins."""
    p = np.asarray(probs, float)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1) + 1
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(bins):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_table(
    probs: np.ndarray, y_true: Sequence[int], n_bins: int = 10
) -> list[dict[str, float | int | None]]:
    """Per equal-width bin of max P: bin edges, count, mean confidence and accuracy.

    Same binning as ``ece``. Empty bins have ``confidence`` and ``accuracy`` None.
    """
    p = np.asarray(probs, float)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) + 1 == np.asarray(y_true)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out: list[dict[str, float | int | None]] = []
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        out.append(
            {
                "low": float(lo),
                "high": float(hi),
                "count": int(m.sum()),
                "confidence": float(conf[m].mean()) if m.any() else None,
                "accuracy": float(correct[m].mean()) if m.any() else None,
            }
        )
    return out


def cluster_bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    groups: Sequence[str],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile CI for ``stat(indices)`` resampling **groups** (sessions), then all their rows.

    ``stat`` receives an integer index array into the original rows.
    """
    g = np.asarray(groups)
    uniq = np.unique(g)
    rows_by_group = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_group[u] for u in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    point = stat(np.arange(len(g)))
    if not vals:
        return {"point": point, "low": float("nan"), "high": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "low": float(lo), "high": float(hi), "n_boot": len(vals)}
