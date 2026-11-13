"""Grader evaluation orchestration (EVALUATION G1, G3, G5). TASKS M3.

``grade_metrics`` (pure) computes the per-model block used in ladder.json.

``run(predictions_dir, labels_csv, split_json, synthetic, ...)`` loads every
``<predictions_dir>/<model>.csv`` (columns ``PRED_COLUMNS``), joins the **test** labels (latest
normal-mode label per window), and writes through ``report.results.write_result``:

- ``grader/ladder.json`` (G1): exact, within-1 and QWK with session-bootstrap CIs per model, in
  ladder order, plus ``best`` (highest QWK point). Each model also gets ``confusion``: 5x5
  counts, rows = true grade 1..5, columns = predicted (``viz.plots.confusion_chart``). An
  optional ``<model>.meta.json`` next to the predictions (e.g. the chosen LightGBM grid point)
  is copied in as ``selection``.
- ``grader/tags.json`` (G3): macro-F1 and per-tag F1 per model.
- ``grader/calibration.json`` (G5): ECE, Brier and the reliability table for models that output
  probabilities (``p1..p5`` filled); the rest are listed under ``no_probabilities``.

- ``grader/verifier.json`` (G4): for models whose ``<model>.meta.json`` says
  ``generates_cues`` (the LLM), the cue-verifier pass rate with a session-bootstrap CI and a
  histogram of failure codes (``llm.verifier.FAILURE_CODES``). Needs ``windows`` (window_id →
  windows.parquet row) and ``features_cfg``; models without cues are listed under ``no_cues``.
- ``grader/consistency.json`` (G6): for models with ``<model>.consistency.json`` (written by
  ``gaitlab predict``): the share of windows whose grade is identical across the repeated
  scoring runs, and whose tags / cue are identical across the sampled generations.

verifier.json and consistency.json are written only when some model has cues / repeats.
With ``synthetic=True`` the files go to ``results/synthetic/grader/`` and are never reported.
``synthetic=False`` is refused when any label or prediction is synthetic. Every model must cover
exactly the same test windows.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gaitlab.eval import metrics as m

LADDER_ORDER = (
    "rules",
    "logreg",
    "lgbm",
    "gemma3_1b_zeroshot",
    "gemma3_1b_lora",
    "gemma3_270m_lora",
)
PRED_COLUMNS = ["window_id", "pred_grade", "p1", "p2", "p3", "p4", "p5", "tags", "cue", "synthetic"]


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
        "confusion": {
            "grades": [int(g) for g in m.GRADES],
            "counts": m.confusion_counts(yt, yp).tolist(),  # rows: true, columns: predicted
        },
    }
    if probs is not None:
        out["ece"] = m.ece(probs, yt)
        out["brier"] = m.brier_multiclass(probs, yt)
    return out


def _tag_set(s: object) -> set[str]:
    return {t for t in str(s if isinstance(s, str) else "").split(";") if t}


def _ordered(names: Sequence[str]) -> list[str]:
    known = [n for n in LADDER_ORDER if n in names]
    return known + sorted(n for n in names if n not in LADDER_ORDER)


def _truthy(v: object) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1")


def run(
    predictions_dir: str,
    labels_csv: str,
    split_json: str,
    synthetic: bool,
    *,
    tag_vocab: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
    config: dict[str, Any] | None = None,
    results_root: str = "results",
    windows: Mapping[str, Mapping[str, Any]] | None = None,
    features_cfg=None,
) -> dict[str, Path]:
    """Score every model's test predictions; write ladder, tags, calibration and, when an LLM
    is present, verifier and consistency results.

    Returns {"ladder" | "tags" | "calibration" [| "verifier" | "consistency"]: written path}.
    """
    import pandas as pd

    from gaitlab.data.sessions import latest_normal_labels, load_labels
    from gaitlab.report.results import manifest_hash, write_result

    split = json.loads(Path(split_json).read_text())
    label_rows = load_labels(labels_csv)
    labels = latest_normal_labels(label_rows)
    test_ids = sorted(w for w in labels if split.get(w.split(":")[0]) == "test")
    if not test_ids:
        raise ValueError("no labeled windows in test sessions")
    pred_files = sorted(Path(predictions_dir).glob("*.csv"))
    if not pred_files:
        raise ValueError(f"no prediction files in {predictions_dir}")

    preds: dict[str, pd.DataFrame] = {}
    for p in pred_files:
        df = pd.read_csv(p, dtype={"window_id": str, "tags": str, "cue": str})
        if missing := set(PRED_COLUMNS) - set(df.columns):
            raise ValueError(f"{p.name}: missing columns {sorted(missing)}")
        got = set(df["window_id"])
        if got != set(test_ids):
            raise ValueError(
                f"{p.stem}: predictions cover {len(got & set(test_ids))} of {len(test_ids)} test "
                f"windows (+{len(got - set(test_ids))} others); every model needs the same windows"
            )
        preds[p.stem] = df.set_index("window_id").loc[test_ids]

    data_synthetic = any(r.rater == "synthetic" for r in label_rows) or any(
        _truthy(v) for df in preds.values() for v in df["synthetic"]
    )
    if data_synthetic and not synthetic:
        raise ValueError("labels or predictions are synthetic; run with synthetic=True")

    y = np.array([labels[w].grade for w in test_ids])
    groups = [w.split(":")[0] for w in test_ids]
    true_tags = [_tag_set(labels[w].tags) for w in test_ids]
    order = _ordered(list(preds))
    pcols = [f"p{k}" for k in range(1, 6)]

    ladder: dict[str, Any] = {}
    tags: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    no_probs: list[str] = []
    metas = {}
    for name in order:
        mp = Path(predictions_dir) / f"{name}.meta.json"
        metas[name] = json.loads(mp.read_text()) if mp.exists() else None
    cue_models = [n for n in order if (metas[n] or {}).get("generates_cues")]
    if cue_models and (windows is None or features_cfg is None):
        raise ValueError(f"{cue_models} generate cues: the verifier needs windows and features_cfg")
    for name in order:
        df = preds[name]
        block = grade_metrics(
            y, df["pred_grade"].astype(int).to_numpy(), groups, None, n_boot, seed
        )
        if metas[name] is not None:
            block["selection"] = metas[name]
        ladder[name] = block
        tags[name] = m.macro_f1_multilabel(true_tags, [_tag_set(t) for t in df["tags"]], tag_vocab)
        probs = df[pcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if np.isfinite(probs).all():
            calibration[name] = {
                "ece": m.ece(probs, y),
                "brier": m.brier_multiclass(probs, y),
                "reliability": m.reliability_table(probs, y),
            }
        else:
            no_probs.append(name)

    verifier = {
        n: verifier_block(test_ids, groups, preds[n], windows, features_cfg, n_boot, seed)  # type: ignore[arg-type]
        for n in cue_models
    }
    cons_files = {n: Path(predictions_dir) / f"{n}.consistency.json" for n in order}
    consistency = {
        n: consistency_block(test_ids, groups, json.loads(p.read_text()), n, n_boot, seed)
        for n, p in cons_files.items()
        if p.exists()
    }

    best = max(order, key=lambda n: ladder[n]["qwk"]["point"])
    extra_inputs = [p for p in cons_files.values() if p.exists()]
    common = {
        "config": {"n_boot": n_boot, "seed": seed, "models": order, **(config or {})},
        "data_manifest": manifest_hash([*pred_files, *extra_inputs, labels_csv, split_json]),
        "synthetic": synthetic,
        "root": results_root,
    }
    base = {"split": "test", "n_windows": len(test_ids), "n_sessions": len(set(groups))}
    out: dict[str, Path] = {}
    if verifier:
        no_cues = [n for n in order if n not in verifier]
        out["verifier"] = write_result(
            "grader", "verifier", {**base, "models": verifier, "no_cues": no_cues}, **common
        )
    if consistency:
        out["consistency"] = write_result(
            "grader", "consistency", {**base, "models": consistency}, **common
        )
    return out | {
        "ladder": write_result(
            "grader", "ladder",
            {**base, "order": order, "models": ladder,
             "best": {"model": best, "qwk": float(ladder[best]["qwk"]["point"])}},
            **common,
        ),
        "tags": write_result("grader", "tags", {**base, "models": tags}, **common),
        "calibration": write_result(
            "grader", "calibration",
            {**base, "models": calibration, "no_probabilities": no_probs, "ece_bins": 10},
            **common,
        ),
    }  # fmt: skip


def _rate_ci(flags: Sequence[bool], groups: Sequence[str], n_boot: int, seed: int) -> dict:
    a = np.asarray(flags, dtype=float)
    return m.cluster_bootstrap_ci(lambda i: float(a[i].mean()), groups, n_boot, seed=seed)


def verifier_block(
    test_ids: Sequence[str],
    groups: Sequence[str],
    df,
    windows: Mapping[str, Mapping[str, Any]],
    features_cfg,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """G4 for one model: pass rate (session-bootstrap CI) and failure-code counts."""
    from gaitlab.llm.verifier import verify

    passed: list[bool] = []
    failures: dict[str, int] = {}
    for wid in test_ids:
        if wid not in windows:
            raise ValueError(f"verifier: window {wid} missing from windows")
        r = verify(
            windows[wid], sorted(_tag_set(df.at[wid, "tags"])),
            df.at[wid, "cue"] if isinstance(df.at[wid, "cue"], str) else "", features_cfg,
        )  # fmt: skip
        passed.append(r.passed)
        for c in r.codes:
            failures[c] = failures.get(c, 0) + 1
    return {
        "n_windows": len(passed),
        "n_passed": int(sum(passed)),
        "pass_rate": _rate_ci(passed, groups, n_boot, seed),
        "failures": dict(sorted(failures.items())),
    }


def consistency_block(
    test_ids: Sequence[str],
    groups: Sequence[str],
    payload: Mapping[str, Any],
    name: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """G6 for one model from its ``<model>.consistency.json``."""
    reps = payload["windows"]
    if set(reps) != set(test_ids):
        raise ValueError(f"{name}.consistency.json covers {len(set(reps) & set(test_ids))} of "
                         f"{len(test_ids)} test windows")  # fmt: skip

    def same(xs: Sequence) -> bool:
        return all(x == xs[0] for x in xs)

    grade = [same(reps[w]["grades"]) for w in test_ids]
    tags = [same([sorted(t) for t in reps[w]["tags"]]) for w in test_ids]
    cue = [same(reps[w]["cues"]) for w in test_ids]
    return {
        "runs": int(payload["runs"]),
        "temperature": float(payload["temperature"]),
        "grade_identical": _rate_ci(grade, groups, n_boot, seed),
        "tags_identical_sampled": _rate_ci(tags, groups, n_boot, seed),
        "cue_identical_sampled": _rate_ci(cue, groups, n_boot, seed),
        "distinct_cues_mean": float(np.mean([len(set(reps[w]["cues"])) for w in test_ids])),
    }
