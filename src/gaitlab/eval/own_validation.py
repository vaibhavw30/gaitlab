"""Validation on own sessions: V5 directions, V6 stability, V7 backend agreement (EVALUATION §1).

Inputs are the per-backend keypoint files ``<sessions dir>/interim/<session>/<clip>.<backend>.npz``
written by ``gaitlab extract``. Strides and windows are recomputed in memory with the same code
as ``gaitlab process`` (``data.analysis``), so the backend behind every number is known and two
backends never overwrite each other's processed parquet.

- **V5** ``validation/directions.json``: for each (condition, feature) in
  ``EXPECTED_DIRECTIONS`` (EVALUATION V5; Heiderscheit et al. 2011), the per-session difference
  median(condition windows) − median(natural windows) of ``<feature>_med``, averaged over the
  sessions that have both clips, with a percentile CI from resampling sessions. ``sign_ok``: the
  point estimate has the expected sign. ``ci_on_expected_side``: the whole CI does.
- **V6** ``validation/stability.json``: ICC(2,1) per feature. Targets are (session, condition)
  cells; repeats are each cell's first ``k`` windows in time order (finite values only), with
  ``k`` the smallest window count among cells that have at least 2.
- **V7** ``validation/backend_agreement.json`` (only with a second backend): near-side strides of
  the two backends matched one-to-one within (session, clip, side) by nearest initial contact,
  accepted when closer than half the first backend's stride time; then Bland-Altman per feature
  (first − second). The matched pairs are written to a parquet under ``data/`` (git-ignored; a
  per-stride table is too large to commit) whose sha256 is in the result, for
  ``viz.plots.bland_altman_chart``.

Results go through ``report.results.write_result``; synthetic keypoints → ``results/synthetic/``.
No thresholds are involved: V5 reports signs and sizes, not pass/fail against a cut-off.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gaitlab.eval import metrics as m
from gaitlab.types import FEATURE_FIELDS

EXPECTED_DIRECTIONS: dict[str, dict[str, int]] = {
    "metronome": {"cadence_spm": 1, "overstride_L": -1, "vert_osc_L": -1},
    "overstride": {"overstride_L": 1, "shin_angle_deg": 1},
}
"""condition → feature → expected sign of (condition − natural). EVALUATION V5."""

MATCH_FRACTION = 0.5
"""Strides match when their ICs are closer than this fraction of a stride: never a neighbour."""


def _num(x: float) -> float | None:
    """JSON-safe float: NaN and inf become None."""
    x = float(x)
    return x if np.isfinite(x) else None


# --- V5 ---------------------------------------------------------------------------------------


def session_differences(windows, condition: str, feature: str) -> dict[str, float]:
    """Per session with both clips: median(condition) − median(natural) of ``<feature>_med``."""
    col = f"{feature}_med"
    out = {}
    for sid, grp in windows.groupby("session_id", sort=True):
        cond = grp.loc[grp["condition"] == condition, col].to_numpy(float)
        nat = grp.loc[grp["condition"] == "natural", col].to_numpy(float)
        cond, nat = cond[np.isfinite(cond)], nat[np.isfinite(nat)]
        if cond.size and nat.size:
            out[str(sid)] = float(np.median(cond) - np.median(nat))
    return out


def direction_check(
    windows, condition: str, feature: str, expected: int, n_boot: int, seed: int
) -> dict[str, Any]:
    """Sign and size of condition − natural with a session-bootstrap CI (EVALUATION V5)."""
    diffs = session_differences(windows, condition, feature)
    sessions = sorted(diffs)
    in_pairs = windows["session_id"].astype(str).isin(sessions)
    n_windows = {
        "condition": int((in_pairs & (windows["condition"] == condition)).sum()),
        "natural": int((in_pairs & (windows["condition"] == "natural")).sum()),
    }
    base = {"expected_sign": expected, "n_sessions": len(sessions), "n_windows": n_windows}
    if not sessions:
        return {**base, "diff": None, "sign_ok": None, "ci_on_expected_side": None}
    vals = np.array([diffs[s] for s in sessions])
    ci = m.cluster_bootstrap_ci(lambda i: float(np.mean(vals[i])), sessions, n_boot, seed=seed)
    point, low, high = ci["point"], ci["low"], ci["high"]
    on_side = bool(low > 0) if expected > 0 else bool(high < 0)
    return {
        **base,
        "diff": {
            "point": _num(point),
            "low": _num(low),
            "high": _num(high),
            "n_boot": ci["n_boot"],
        },
        "sign_ok": bool(np.sign(point) == expected),
        "ci_on_expected_side": on_side if np.isfinite([low, high]).all() else None,
    }


def directions(windows, n_boot: int, seed: int) -> dict[str, Any]:
    """Every expected direction; conditions with no windows are listed under ``missing``."""
    present = set(windows["condition"])
    conds = {}
    for cond, feats in EXPECTED_DIRECTIONS.items():
        if cond not in present or "natural" not in present:
            continue
        conds[cond] = {f: direction_check(windows, cond, f, s, n_boot, seed)
                       for f, s in feats.items()}  # fmt: skip
    missing = sorted(set(EXPECTED_DIRECTIONS) - set(conds))
    return {"conditions": conds, "missing_conditions": missing}


# --- V6 ---------------------------------------------------------------------------------------


def stability_matrix(
    windows, feature: str
) -> tuple[np.ndarray, list[tuple[str, str]], list[tuple[str, str]]]:
    """(cells × k) ratings, the cells used and the cells dropped (fewer than 2 finite windows)."""
    col = f"{feature}_med"
    series: dict[tuple[str, str], np.ndarray] = {}
    for (sid, cond), grp in windows.groupby(["session_id", "condition"], sort=True):
        v = grp.sort_values("start_stride")[col].to_numpy(float)
        series[(str(sid), str(cond))] = v[np.isfinite(v)]
    used = [c for c, v in series.items() if v.size >= 2]
    dropped = [c for c, v in series.items() if v.size < 2]
    if not used:
        return np.empty((0, 0)), [], dropped
    k = min(series[c].size for c in used)
    return np.stack([series[c][:k] for c in used]), used, dropped


def stability(windows, features: Sequence[str]) -> dict[str, dict[str, Any]]:
    """ICC(2,1) per feature across windows of the same session and condition (EVALUATION V6)."""
    out = {}
    for f in features:
        ratings, used, dropped = stability_matrix(windows, f)
        row: dict[str, Any] = {"n_cells": len(used), "k": int(ratings.shape[1]) if used else 0,
                               "dropped_cells": [list(c) for c in dropped]}  # fmt: skip
        if len(used) < 2:
            row |= {"icc": None, "reason": "needs at least 2 cells (session x condition) with "
                    "2 or more windows"}  # fmt: skip
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                icc = _num(m.icc_2_1(ratings))
            row |= {"icc": icc} if icc is not None else {"icc": None, "reason": "no variance"}
        out[f] = row
    return out


# --- V7 ---------------------------------------------------------------------------------------


def match_strides(a, b) -> list[tuple[int, int]]:
    """One-to-one (row in a, row in b) by nearest IC within (session, clip, side).

    A pair is allowed when |t_ic difference| < ``MATCH_FRACTION`` × a's stride time; pairs are
    taken closest first.
    """
    pairs = []
    keys = ["session_id", "clip_id", "side"]
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    b_groups = {k: g for k, g in b.groupby(keys, sort=False)}
    for k, ga in a.groupby(keys, sort=False):
        gb = b_groups.get(k)
        if gb is None:
            continue
        ta, tb = ga["t_ic"].to_numpy(float), gb["t_ic"].to_numpy(float)
        tol = MATCH_FRACTION * ga["stride_time_s"].to_numpy(float)
        dt = np.abs(ta[:, None] - tb[None, :])
        cand = [(dt[i, j], i, j) for i in range(len(ta)) for j in range(len(tb))
                if np.isfinite(dt[i, j]) and dt[i, j] < tol[i]]  # fmt: skip
        used_a, used_b = set(), set()
        for _, i, j in sorted(cand):
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            pairs.append((int(ga.index[i]), int(gb.index[j])))
    return pairs


def paired_strides(a, b, features: Sequence[str]):
    """Matched strides side by side: keys, ``t_ic_a`` / ``t_ic_b`` and ``<feature>_a`` / ``_b``."""
    import pandas as pd

    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    pairs = match_strides(a, b)
    ia = [p[0] for p in pairs]
    ib = [p[1] for p in pairs]
    cols = {k: a[k].to_numpy()[ia] for k in ("session_id", "clip_id", "side")}
    cols |= {"t_ic_a": a["t_ic"].to_numpy(float)[ia], "t_ic_b": b["t_ic"].to_numpy(float)[ib]}
    for f in features:
        cols[f"{f}_a"] = a[f].to_numpy(float)[ia]
        cols[f"{f}_b"] = b[f].to_numpy(float)[ib]
    return pd.DataFrame(cols)


def agreement_summary(pairs, features: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Bland-Altman (a − b) per feature over the pairs where both values are finite."""
    out = {}
    for f in features:
        va, vb = pairs[f"{f}_a"].to_numpy(float), pairs[f"{f}_b"].to_numpy(float)
        ok = np.isfinite(va) & np.isfinite(vb)
        if not ok.any():
            out[f] = {"bias": None, "loa_low": None, "loa_high": None, "n": 0}
            continue
        ba = m.bland_altman(va[ok], vb[ok])
        out[f] = {k: (int(v) if k == "n" else _num(v)) for k, v in ba.items()}
    return out


def backend_agreement(a, b, features: Sequence[str]) -> tuple[dict[str, Any], Any]:
    """(summary, pairs table): Bland-Altman per feature on one-to-one matched strides."""
    pairs = paired_strides(a, b, features)
    summary = {
        "n_pairs": len(pairs),
        "unmatched": {"a": len(a) - len(pairs), "b": len(b) - len(pairs)},
        "features": agreement_summary(pairs, features),
    }
    return summary, pairs


# --- orchestration ----------------------------------------------------------------------------


def _analyze(rows_by_session: dict[str, list], base: Path, backend: str, cfg):
    import pandas as pd

    from gaitlab.data.analysis import analyze_session

    strides, windows, flags, npz = [], [], set(), []
    for sid, rows in rows_by_session.items():
        r = analyze_session(rows, base / "interim" / sid, backend, cfg)
        strides += r.strides
        windows += r.windows
        flags |= r.synthetic
        npz += r.npz
    return pd.DataFrame(strides), pd.DataFrame(windows), flags, npz


def run(
    sessions_csv: str,
    *,
    backend: str,
    synthetic: bool,
    other: str | None = None,
    config_path: str = "configs/default.yaml",
    n_boot: int | None = None,
    session_ids: Sequence[str] | None = None,
    results_root: str = "results",
    pairs_dir: str | None = None,
) -> dict[str, Path]:
    """Write V5 and V6 for ``backend`` and, with ``other``, V7 (``backend`` − ``other``).

    V7's matched stride pairs go to ``<pairs_dir>/backend_pairs.<backend>-<other>.parquet``
    (default ``<sessions dir>/processed/``, under ``data/``); the result JSON keeps the summary,
    the file name and its sha256.

    Returns {"directions" | "stability" [| "backend_agreement" | "pairs"]: written path}.
    """
    from gaitlab.config import load_pipeline_config
    from gaitlab.data.sessions import load_sessions
    from gaitlab.report.results import manifest_hash, write_result

    cfg = load_pipeline_config(config_path)
    n_boot = n_boot or cfg.bootstrap.n
    by_session: dict[str, list] = {}
    for r in load_sessions(sessions_csv):
        if session_ids is None or r.session_id in session_ids:
            by_session.setdefault(r.session_id, []).append(r)
    if not by_session:
        raise ValueError(f"no sessions to validate in {sessions_csv}")
    base = Path(sessions_csv).parent

    strides, windows, flags, npz = _analyze(by_session, base, backend, cfg)
    if other is not None:
        o_strides, _, o_flags, o_npz = _analyze(by_session, base, other, cfg)
        flags, npz = flags | o_flags, npz + o_npz
    if len(flags) > 1:
        raise ValueError("these sessions mix synthetic and real keypoints")
    if flags == {True} and not synthetic:
        raise ValueError("keypoints are synthetic; run with synthetic=True (--synthetic)")
    if windows.empty:
        raise ValueError("no windows: every clip is shorter than one window or below quality_min")

    common = {
        "config": {
            "pipeline": cfg.model_dump(mode="json"),
            "backend": backend,
            "other": other,
            "n_boot": n_boot,
            "seed": cfg.seed,
        },
        "data_manifest": manifest_hash([*npz, sessions_csv]),
        "synthetic": synthetic or flags == {True},
        "root": results_root,
    }
    head = {"backend": backend, "n_sessions": len(by_session),
            "sessions": sorted(by_session), "n_windows": len(windows)}  # fmt: skip
    out = {
        "directions": write_result(
            "validation", "directions",
            {**head, "bootstrap": "sessions", **directions(windows, n_boot, cfg.seed)},
            notes="condition - natural of per-session window medians; mean over sessions",
            **common,
        ),
        "stability": write_result(
            "validation", "stability",
            {**head, "features": stability(windows, FEATURE_FIELDS)},
            notes="ICC(2,1): targets = session x condition cells, repeats = windows in time order",
            **common,
        ),
    }  # fmt: skip
    if other is not None:
        agree, pairs = backend_agreement(strides, o_strides, FEATURE_FIELDS)
        pairs["synthetic"] = common["synthetic"]
        pairs_path = Path(pairs_dir) if pairs_dir else base / "processed"
        pairs_path = pairs_path / f"backend_pairs.{backend}-{other}.parquet"
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_parquet(pairs_path, index=False)
        out["backend_agreement"] = write_result(
            "validation", "backend_agreement",
            {"backends": [backend, other], "n_sessions": len(by_session),
             "sessions": sorted(by_session), "difference": f"{backend} - {other}", **agree,
             "pairs_file": pairs_path.name, "pairs_sha256": _sha256(pairs_path)},
            notes="near-side strides matched by nearest IC within half a stride; per-pair "
                  "values in pairs_file (under data/, not committed)",
            **common,
        )  # fmt: skip
        out["pairs"] = pairs_path
    return out


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
