"""Tabular baselines: logistic regression and LightGBM (TECHNICAL_PLAN §7.1). TASKS M2.

Contract:
- ``feature_matrix(rows, feature_order) -> np.ndarray``: ``<f>_med`` and ``<f>_iqr`` columns in
  fixed order; NaN kept (LightGBM handles it; LR uses median imputation fitted on train only).
- ``fit_logreg(X, y, seed) / fit_lgbm(X, y, seed)``: multinomial LR on standardized features;
  LightGBM multiclass with a small fixed grid (num_leaves ∈ {7, 15}, n_estimators ∈ {100, 300},
  learning_rate 0.05, min_child_samples 5) chosen by leave-one-session-out CV on train+dev
  (``data.splits.leave_one_session_out``), scored by QWK.
- ``predict_proba(model, X) -> (N, 5)`` with columns for grades 1..5 even if a grade is
  absent from training (fill with 0 and renormalize).
- Tags for baselines: one-vs-rest LR per tag on the same features (macro-F1 comparable to LLM).
Imports of sklearn/lightgbm stay inside functions so the core installs without the ml extra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def feature_matrix(rows: Sequence[Mapping[str, float]], feature_order: Sequence[str]) -> np.ndarray:
    cols = [f"{f}_{s}" for f in feature_order for s in ("med", "iqr")]
    return np.array([[float(r.get(c, np.nan)) for c in cols] for r in rows], dtype=np.float64)


def fit_logreg(X: np.ndarray, y: np.ndarray, seed: int = 0):
    raise NotImplementedError("TASKS.md M2: baselines")


def fit_lgbm(X: np.ndarray, y: np.ndarray, seed: int = 0):
    raise NotImplementedError("TASKS.md M2: baselines")
