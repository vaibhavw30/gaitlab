"""Grader evaluation orchestration (EVALUATION G1–G6). TASKS M3.

``grade_metrics`` (implemented, pure) computes the per-model block used in ladder.json.
``run(predictions_dir, labels_csv, split_json, synthetic)`` (to implement) loads each model's
predictions (``window_id, pred_grade, p1..p5, tags, cue``), joins test labels, computes
``grade_metrics`` per model, tag F1, verifier pass rate, calibration and consistency, and writes
``results/grader/*.json`` (or results/synthetic/grader) via ``report.results.write_result``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gaitlab.eval import metrics as m


def grade_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str],
    probs: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    out: dict[str, object] = {
        "n_windows": len(yt),
        "n_sessions": len(set(groups)),
        "exact": m.cluster_bootstrap_ci(
            lambda i: m.exact_agreement(yt[i], yp[i]), groups, n_boot, seed=seed
        ),
        "within_1": m.cluster_bootstrap_ci(
            lambda i: m.within_one(yt[i], yp[i]), groups, n_boot, seed=seed
        ),
        "qwk": m.cluster_bootstrap_ci(
            lambda i: m.quadratic_weighted_kappa(yt[i], yp[i]), groups, n_boot, seed=seed
        ),
    }
    if probs is not None:
        out["ece"] = m.ece(probs, yt)
        out["brier"] = m.brier_multiclass(probs, yt)
    return out


def run(predictions_dir: str, labels_csv: str, split_json: str, synthetic: bool) -> None:
    raise NotImplementedError("TASKS.md M3: eval orchestration")
