"""Tabular baselines: logistic regression and LightGBM (TECHNICAL_PLAN §7.1). TASKS M2.

- ``feature_matrix(rows, feature_order)``: ``<f>_med`` and ``<f>_iqr`` columns in fixed order;
  NaN kept (LightGBM handles it; LR imputes the median, fitted on the training rows only).
- ``fit_logreg``: multinomial LR on standardized features (lbfgs, C = 1).
- ``fit_lgbm``: LightGBM with a small fixed grid (``LGBM_GRID``; learning_rate 0.05,
  min_child_samples 5), the grid point chosen by leave-one-session-out QWK when ``groups`` is
  given. Per-point CV scores are kept in ``GradeModel.cv_qwk``.
- ``predict_proba(model, X) -> (N, 5)``: columns for grades 1..5 even if a grade is absent from
  training (filled with 0, rows renormalized). A model trained on one grade predicts it with P=1.
- ``loso_proba``: out-of-fold probabilities, one fold per held-out session.
- Tags: one-vs-rest LR per tag on the same features (``fit_tags`` / ``predict_tags``), shared by
  both baselines so their macro-F1 is comparable with the LLM's.
- ``baseline_predictions``: fit on train + dev sessions, predict test → rows in the predictions
  schema read by ``eval.run_eval``.
sklearn and lightgbm are imported inside functions so the core installs without the ml extra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GRADES = (1, 2, 3, 4, 5)
LGBM_FIXED = {"learning_rate": 0.05, "min_child_samples": 5}
LGBM_GRID: list[dict[str, int]] = [
    {"num_leaves": nl, "n_estimators": ne} for nl in (7, 15) for ne in (100, 300)
]


def feature_matrix(rows: Sequence[Mapping[str, float]], feature_order: Sequence[str]) -> np.ndarray:
    cols = feature_columns(feature_order)
    return np.array([[_num(r.get(c)) for c in cols] for r in rows], dtype=np.float64).reshape(
        len(rows), len(cols)
    )


def feature_columns(feature_order: Sequence[str]) -> list[str]:
    """``<f>_med``, ``<f>_iqr`` per feature, in order."""
    return [f"{f}_{s}" for f in feature_order for s in ("med", "iqr")]


def _num(v: object) -> float:
    return float("nan") if v is None else float(v)  # type: ignore[arg-type]


def grid_key(params: Mapping[str, int]) -> str:
    return ",".join(f"{k}={params[k]}" for k in sorted(params))


@dataclass
class GradeModel:
    """A fitted grade classifier. ``estimator`` is None when training had a single grade."""

    kind: str
    estimator: Any
    classes: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)
    cv_qwk: dict[str, float] = field(default_factory=dict)


def _single_grade(kind: str, y: np.ndarray, params: dict | None = None) -> GradeModel | None:
    classes = np.unique(y)
    return GradeModel(kind, None, classes, params or {}) if len(classes) == 1 else None


def fit_logreg(X: np.ndarray, y: np.ndarray, seed: int = 0) -> GradeModel:
    """Multinomial logistic regression on median-imputed, standardized features."""
    y = np.asarray(y, dtype=int)
    if (m := _single_grade("logreg", y)) is not None:
        return m
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    est = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
    )
    est.fit(X, y)
    return GradeModel("logreg", est, np.asarray(est.classes_, dtype=int), {"C": 1.0})


def _fit_lgbm_params(X: np.ndarray, y: np.ndarray, params: dict, seed: int) -> GradeModel:
    y = np.asarray(y, dtype=int)
    if (m := _single_grade("lgbm", y, params)) is not None:
        return m
    from lightgbm import LGBMClassifier

    est = LGBMClassifier(**LGBM_FIXED, **params, random_state=seed, n_jobs=1, verbose=-1)
    est.fit(X, y)
    return GradeModel("lgbm", est, np.asarray(est.classes_, dtype=int), dict(params))


def fit_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str] | None = None,
    seed: int = 0,
    grid: Sequence[dict[str, int]] = tuple(LGBM_GRID),
) -> GradeModel:
    """LightGBM; with ``groups`` (≥ 2 sessions) the grid point is chosen by LOSO QWK."""
    from gaitlab.eval.metrics import quadratic_weighted_kappa

    y = np.asarray(y, dtype=int)
    scores: dict[str, float] = {}
    best = dict(grid[0])
    if groups is not None and len(set(groups)) >= 2 and len(np.unique(y)) > 1:
        for params in grid:
            oof = loso_proba(
                lambda Xtr, ytr, _idx, p=params: _fit_lgbm_params(Xtr, ytr, p, seed), X, y, groups
            )
            scores[grid_key(params)] = quadratic_weighted_kappa(y, oof.argmax(axis=1) + 1)
        # First grid point wins ties (grid order is smallest model first).
        best = dict(max(grid, key=lambda p: scores[grid_key(p)]))
    model = _fit_lgbm_params(X, y, best, seed)
    model.cv_qwk = scores
    return model


def predict_proba(model: GradeModel, X: np.ndarray) -> np.ndarray:
    """(N, 5) probabilities for grades 1..5; grades unseen in training get 0."""
    out = np.zeros((len(X), len(GRADES)), dtype=np.float64)
    cols = np.asarray(model.classes, dtype=int) - 1
    if model.estimator is None:
        out[:, cols[0]] = 1.0
        return out
    out[:, cols] = model.estimator.predict_proba(X)
    s = out.sum(axis=1, keepdims=True)
    return out / np.where(s > 0, s, 1.0)


def predict_grade(model: GradeModel, X: np.ndarray) -> np.ndarray:
    """Argmax grade (1..5)."""
    return predict_proba(model, X).argmax(axis=1) + 1


FitFn = Callable[[np.ndarray, np.ndarray, np.ndarray], GradeModel]
"""(X_train, y_train, train row indices) → fitted model."""


def loso_proba(fit: FitFn, X: np.ndarray, y: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    """Out-of-fold (N, 5) probabilities, holding out one session at a time."""
    from gaitlab.data.splits import leave_one_session_out

    y = np.asarray(y, dtype=int)
    oof = np.zeros((len(X), len(GRADES)))
    for train, val in leave_one_session_out(list(groups)):
        tr = np.asarray(train, dtype=int)
        model = fit(X[tr], y[tr], tr)
        oof[val] = predict_proba(model, X[val])
    return oof


@dataclass
class TagModels:
    """One-vs-rest classifiers per tag; a bool stands in where training had one class."""

    vocab: list[str]
    models: dict[str, Any]


def fit_tags(
    X: np.ndarray, tag_sets: Sequence[set[str]], vocab: Sequence[str], seed: int = 0
) -> TagModels:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models: dict[str, Any] = {}
    for tag in vocab:
        yt = np.array([tag in s for s in tag_sets])
        if yt.all() or not yt.any():
            models[tag] = bool(yt.all()) if len(yt) else False
            continue
        est = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        )
        models[tag] = est.fit(X, yt)
    return TagModels(list(vocab), models)


def predict_tags(tm: TagModels, X: np.ndarray, threshold: float = 0.5) -> list[set[str]]:
    out: list[set[str]] = [set() for _ in range(len(X))]
    for tag in tm.vocab:
        m = tm.models[tag]
        hit = np.full(len(X), m) if isinstance(m, bool) else m.predict_proba(X)[:, 1] >= threshold
        for i in np.flatnonzero(hit):
            out[i].add(tag)
    return out


def _pred_row(wid: str, grade: int, probs: np.ndarray | None, tags: set[str] | list[str],
              vocab: Sequence[str], synthetic: bool) -> dict[str, Any]:  # fmt: skip
    row: dict[str, Any] = {"window_id": wid, "pred_grade": int(grade)}
    for k in GRADES:
        row[f"p{k}"] = None if probs is None else float(probs[k - 1])
    row["tags"] = ";".join(t for t in vocab if t in tags)
    row["cue"] = ""
    row["synthetic"] = synthetic
    return row


def baseline_predictions(
    windows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, str],
    features_cfg,
    feature_order: Sequence[str],
    synthetic: bool,
    seed: int = 0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rules, LR and LightGBM predictions for the **test** sessions.

    LR, LightGBM and the tag models are fit on labeled train + dev windows only; LightGBM's grid
    is chosen by leave-one-session-out CV there. Rules need ``features_cfg`` thresholds (raises
    ``ThresholdsNotCalibratedError`` while null). Returns (model → prediction rows, metadata).
    """
    from gaitlab.models.rules import rules_predict

    vocab = list(features_cfg.tags)
    labeled = [w for w in windows if str(w["window_id"]) in labels]
    for w in labeled:
        if str(w["session_id"]) not in split:
            raise KeyError(f"session {w['session_id']} missing from split")
    fit_rows = [w for w in labeled if split[str(w["session_id"])] in ("train", "dev")]
    test_rows = [w for w in labeled if split[str(w["session_id"])] == "test"]
    if not fit_rows or not test_rows:
        raise ValueError("need labeled windows in both train+dev and test sessions")

    def _y(rows):
        return np.array([int(labels[str(w["window_id"])]["grade"]) for w in rows])

    def _tags(rows):
        return [
            {t for t in str(labels[str(w["window_id"])].get("tags") or "").split(";") if t}
            for w in rows
        ]

    Xf, Xt = feature_matrix(fit_rows, feature_order), feature_matrix(test_rows, feature_order)
    yf, groups = _y(fit_rows), np.array([str(w["session_id"]) for w in fit_rows])
    tag_model = fit_tags(Xf, _tags(fit_rows), vocab, seed)
    test_tags = predict_tags(tag_model, Xt)
    ids = [str(w["window_id"]) for w in test_rows]

    lr = fit_logreg(Xf, yf, seed)
    lgbm = fit_lgbm(Xf, yf, groups=groups, seed=seed)
    preds: dict[str, list[dict[str, Any]]] = {"rules": []}
    for wid, w in zip(ids, test_rows, strict=True):
        grade, tags = rules_predict(w, features_cfg)
        preds["rules"].append(_pred_row(wid, grade, None, tags, vocab, synthetic))
    for name, model in (("logreg", lr), ("lgbm", lgbm)):
        p = predict_proba(model, Xt)
        preds[name] = [
            _pred_row(wid, int(np.argmax(pi)) + 1, pi, tg, vocab, synthetic)
            for wid, pi, tg in zip(ids, p, test_tags, strict=True)
        ]
    meta = {
        "train_sessions": sorted(set(groups.tolist())),
        "test_sessions": sorted({str(w["session_id"]) for w in test_rows}),
        "n_fit_windows": len(fit_rows),
        "n_test_windows": len(test_rows),
        "feature_columns": feature_columns(feature_order),
        "synthetic": synthetic,
        "logreg": {"params": lr.params},
        "lgbm": {"params": lgbm.params, "cv_qwk": lgbm.cv_qwk},
    }
    return preds, meta
