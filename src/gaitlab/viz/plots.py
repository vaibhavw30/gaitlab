"""Charts (PRD visuals; TASKS M3). All plots read from results/*.json or processed parquet,
never from typed numbers.

Each chart is saved as PNG (2x, 200 dpi) and SVG next to ``out_stem``; real charts go under
``docs/assets/``. Charts from synthetic inputs are titled and watermarked SYNTHETIC and refuse
to write anywhere under ``docs/`` (CLAUDE.md rule 3).

- ``ladder_chart(ladder.json, out_stem, intra_rater_json=None)``: QWK point + 95% CI whisker per
  model in ladder order; a dashed line at the intra-rater QWK when that file is given.
- ``knee_by_condition(strides.parquet paths, out_stem)``: knee flexion across the gait cycle
  (0–100% of stride, the 101-point ``knee_flex_cycle_deg`` column), median line with IQR band
  per condition.
- To do: ``stage_latency``, ``bland_altman``, ``confusion``, ``reliability``.
Title each real chart with the finding, not the topic, once the numbers exist.
matplotlib is imported inside the plotting functions (``viz`` extra).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def ladder_rows(payload: dict[str, Any]) -> tuple[list[tuple[str, float, float, float]], bool]:
    """(model, QWK point, low, high) in ladder order, and whether the result is synthetic."""
    met = payload["metrics"]
    rows = []
    for name in met["order"]:
        q = met["models"][name]["qwk"]
        rows.append((name, float(q["point"]), float(q["low"]), float(q["high"])))
    return rows, bool(payload.get("synthetic"))


def knee_bands(strides) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], bool]:
    """Per condition: (median, 25th, 75th percentile) of the 101-point knee curves, deg.

    ``strides`` is a DataFrame with ``condition`` and ``knee_flex_cycle_deg`` columns. Returns
    the bands and whether any row is synthetic.
    """
    bands = {}
    for cond, grp in strides.groupby("condition", sort=True):
        curves = np.stack([np.asarray(c, dtype=float) for c in grp["knee_flex_cycle_deg"]])
        q25, med, q75 = np.nanpercentile(curves, [25, 50, 75], axis=0)
        bands[str(cond)] = (med, q25, q75)
    synthetic = bool(strides["synthetic"].any()) if "synthetic" in strides else False
    return bands, synthetic


def _check_out(out_stem: Path, synthetic: bool) -> None:
    if synthetic and "docs" in Path(out_stem).resolve().parts:
        raise ValueError(f"synthetic chart must not be written under docs/: {out_stem}")


def _save(fig, out_stem: str | Path, synthetic: bool) -> list[Path]:
    out = Path(out_stem)
    _check_out(out, synthetic)
    out.parent.mkdir(parents=True, exist_ok=True)
    if synthetic:  # below the axes, so it never covers labels; bbox_inches="tight" keeps it
        fig.text(0.99, -0.02, "SYNTHETIC: plumbing only, never report", ha="right", va="top",
                 color="#c0392b", fontsize=8)  # fmt: skip
    paths = [out.with_suffix(".png"), out.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def ladder_chart(
    ladder_json: str | Path, out_stem: str | Path, intra_rater_json: str | Path | None = None
) -> list[Path]:
    """QWK with 95% CI per model (ladder order); dashed line at the intra-rater QWK if given."""
    from matplotlib.figure import Figure

    payload = json.loads(Path(ladder_json).read_text())
    rows, synthetic = ladder_rows(payload)
    _check_out(Path(out_stem), synthetic)
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    x = np.arange(len(rows))
    pts = np.array([r[1] for r in rows])
    err = np.array([[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]])
    ax.errorbar(x, pts, yerr=err, fmt="o", color="#2c3e50", capsize=4)
    if intra_rater_json is not None:
        intra = json.loads(Path(intra_rater_json).read_text())
        synthetic = synthetic or bool(intra.get("synthetic"))
        ax.axhline(float(intra["metrics"]["qwk"]), ls="--", color="#7f8c8d", label="intra-rater")
        ax.legend(frameon=False, loc="lower right")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("QWK (95% CI, session bootstrap)")
    ax.set_ylim(min(0.0, float(np.nanmin([r[2] for r in rows])) - 0.05), 1.05)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(("SYNTHETIC: " if synthetic else "") + "Grader ladder, test sessions")
    return _save(fig, out_stem, synthetic)


def knee_by_condition(strides_paths: Sequence[str | Path], out_stem: str | Path) -> list[Path]:
    """Knee flexion over the gait cycle, median and IQR band per condition."""
    import pandas as pd
    from matplotlib.figure import Figure

    df = pd.concat([pd.read_parquet(p) for p in strides_paths], ignore_index=True)
    bands, synthetic = knee_bands(df)
    _check_out(Path(out_stem), synthetic)
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    pct = np.linspace(0, 100, 101)
    for cond, (med, q25, q75) in bands.items():
        (line,) = ax.plot(pct, med, label=cond, lw=1.6)
        ax.fill_between(pct, q25, q75, color=line.get_color(), alpha=0.2, lw=0)
    ax.set_xlabel("gait cycle, % of stride (IC to next IC)")
    ax.set_ylabel("knee flexion, deg")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(("SYNTHETIC: " if synthetic else "") + "Near-side knee flexion by condition")
    return _save(fig, out_stem, synthetic)
