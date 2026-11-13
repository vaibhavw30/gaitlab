"""Charts (PRD visuals; TASKS M3). All plots read from results/*.json or processed parquet,
never from typed numbers.

Each chart is saved as PNG (2x, 200 dpi) and SVG next to ``out_stem``; real charts go under
``docs/assets/``. Charts from synthetic inputs are titled and watermarked SYNTHETIC and refuse
to write anywhere under ``docs/`` (CLAUDE.md rule 3). Charts from result JSON written on a dirty
git tree also refuse ``docs/`` (rule 1: only ``synthetic: false``, ``git_dirty: false`` is
reportable).

- ``ladder_chart(ladder.json, out_stem, intra_rater_json=None)``: QWK point + 95% CI whisker per
  model in ladder order; a dashed line at the intra-rater QWK when that file is given.
- ``knee_by_condition(strides.parquet paths, out_stem)``: knee flexion across the gait cycle
  (0–100% of stride, the 101-point ``knee_flex_cycle_deg`` column), median line with IQR band
  per condition.
- ``confusion_chart(ladder.json, out_stem, model=None)``: 5x5 grade confusion of one model
  (default: the ladder's ``best``), rows = my grade, columns = predicted. Cell shade = share of
  that row (one-hue blue ramp), cell text = window count; grades absent from the test labels are
  a gray row.
- ``reliability_chart(calibration.json, out_stem, models=None)``: accuracy vs mean confidence per
  non-empty bin (the G5 bins), one line per model that outputs probabilities, a dashed diagonal
  for perfect calibration, and windows per bin underneath. ECE in the legend, read from the file.
- ``bland_altman_chart(backend_agreement.json, pairs.parquet, out_stem, features=None)``: V7,
  one panel per feature: difference (first − second backend) vs mean of the two, per matched
  stride, with the bias line and 95% limits of agreement read from the result. The pairs file
  must match the ``pairs_sha256`` in the result.
- To do: ``stage_latency``.

Colors (new charts): one fixed categorical slot per ladder rung (``series_color``), so a model
keeps its color in every chart whichever models are present; models outside the ladder take
slot 7 or 8.
Several slots are below 3:1 on white, so every series also has its own marker and a legend.
Title each real chart with the finding, not the topic, once the numbers exist.
matplotlib is imported inside the plotting functions (``viz`` extra).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
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


# Categorical slots in ladder order; one-hue blue ramp (light to dark) for magnitudes.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948")  # fmt: skip
MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")
BLUES = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")  # fmt: skip
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, EMPTY = "#e1e0d9", "#c3c2b7", "#f0efec"


@dataclass(frozen=True)
class Provenance:
    """Whether a chart's inputs are synthetic or from a dirty git tree (either: not reportable)."""

    synthetic: bool = False
    dirty: bool = False

    @classmethod
    def of(cls, payload: dict[str, Any]) -> Provenance:
        return cls(bool(payload.get("synthetic")), bool(payload.get("git_dirty")))

    def __or__(self, other: Provenance) -> Provenance:
        return Provenance(self.synthetic or other.synthetic, self.dirty or other.dirty)


def series_color(model: str) -> str:
    """The model's fixed categorical color: its ladder rung; a model outside the ladder gets
    slot 7 or 8 from a hash of its name, so it too keeps its color in every chart."""
    from gaitlab.eval.run_eval import LADDER_ORDER

    if model in LADDER_ORDER:
        return SERIES_LIGHT[LADDER_ORDER.index(model)]
    extra = len(LADDER_ORDER) + int(hashlib.sha256(model.encode()).hexdigest(), 16) % (
        len(SERIES_LIGHT) - len(LADDER_ORDER)
    )
    return SERIES_LIGHT[extra]


def confusion_rows(
    payload: dict[str, Any], model: str | None = None
) -> tuple[np.ndarray, str, Provenance]:
    """(5, 5) counts (rows true 1..5, columns predicted), the model and the file's provenance."""
    met = payload["metrics"]
    model = model or met["best"]["model"]
    if model not in met["models"]:
        raise ValueError(f"{model!r} is not in this ladder file (has {sorted(met['models'])})")
    block = met["models"][model]
    if "confusion" not in block:
        raise ValueError("this ladder file has no confusion counts: rerun `gaitlab eval`")
    counts = np.asarray(block["confusion"]["counts"], dtype=int)
    if counts.shape != (5, 5):
        raise ValueError(f"confusion counts must be 5x5, got {counts.shape}")
    return counts, model, Provenance.of(payload)


def reliability_rows(
    payload: dict[str, Any], models: Sequence[str] | None = None
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], Provenance]:
    """Per model in ladder order: (mean confidence, accuracy, count) of its non-empty bins."""
    from gaitlab.eval.run_eval import _ordered

    met = payload["metrics"]
    have = met["models"]
    if not have:
        raise ValueError("no model in this calibration file outputs probabilities")
    names = _ordered(list(models) if models is not None else list(have))
    if missing := [n for n in names if n not in have]:
        raise ValueError(f"no probabilities (no reliability table) for {missing}")
    rows = {}
    for n in names:
        bins = [b for b in have[n]["reliability"] if b["count"]]
        rows[n] = (
            np.array([b["confidence"] for b in bins], dtype=float),
            np.array([b["accuracy"] for b in bins], dtype=float),
            np.array([b["count"] for b in bins], dtype=int),
        )
    return rows, Provenance.of(payload)


def bland_altman_rows(
    payload: dict[str, Any], pairs, features: Sequence[str] | None = None
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]], Provenance]:
    """Per feature (result order): (mean of a and b, a − b, Bland-Altman summary), finite pairs."""
    summaries = payload["metrics"]["features"]
    names = list(features) if features is not None else list(summaries)
    if missing := [f for f in names if f not in summaries or f"{f}_a" not in pairs]:
        raise ValueError(f"not in this agreement result / pairs file: {missing}")
    rows = {}
    for f in names:
        a, b = pairs[f"{f}_a"].to_numpy(float), pairs[f"{f}_b"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        rows[f] = ((a[ok] + b[ok]) / 2, a[ok] - b[ok], summaries[f])
    prov = Provenance.of(payload)
    if "synthetic" in pairs and bool(pairs["synthetic"].any()):
        prov = prov | Provenance(synthetic=True)
    return rows, prov


def _check_out(out_stem: Path, synthetic: bool, dirty: bool = False) -> None:
    if "docs" not in Path(out_stem).resolve().parts:
        return
    if synthetic:
        raise ValueError(f"synthetic chart must not be written under docs/: {out_stem}")
    if dirty:
        raise ValueError(f"chart from a dirty-tree result must not be written under docs/: "
                         f"{out_stem} (commit, rerun, then plot)")  # fmt: skip


def _style(ax) -> None:
    """Recessive chrome: no top/right spines, muted axes and ticks, secondary-ink labels."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _save(fig, out_stem: str | Path, synthetic: bool, dirty: bool = False) -> list[Path]:
    out = Path(out_stem)
    _check_out(out, synthetic, dirty)
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
    dirty = Provenance.of(payload).dirty
    _check_out(Path(out_stem), synthetic, dirty)
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    x = np.arange(len(rows))
    pts = np.array([r[1] for r in rows])
    err = np.array([[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]])
    ax.errorbar(x, pts, yerr=err, fmt="o", color="#2c3e50", capsize=4)
    if intra_rater_json is not None:
        intra = json.loads(Path(intra_rater_json).read_text())
        synthetic = synthetic or bool(intra.get("synthetic"))
        dirty = dirty or bool(intra.get("git_dirty"))
        ax.axhline(float(intra["metrics"]["qwk"]), ls="--", color="#7f8c8d", label="intra-rater")
        ax.legend(frameon=False, loc="lower right")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("QWK (95% CI, session bootstrap)")
    ax.set_ylim(min(0.0, float(np.nanmin([r[2] for r in rows])) - 0.05), 1.05)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(("SYNTHETIC: " if synthetic else "") + "Grader ladder, test sessions")
    return _save(fig, out_stem, synthetic, dirty)


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


def confusion_chart(
    ladder_json: str | Path, out_stem: str | Path, model: str | None = None
) -> list[Path]:
    """Grade confusion of one model: shade = share of the true-grade row, text = count."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    counts, model, prov = confusion_rows(json.loads(Path(ladder_json).read_text()), model)
    _check_out(Path(out_stem), prov.synthetic, prov.dirty)
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.full(counts.shape, np.nan), where=totals > 0)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES).with_extremes(bad=EMPTY)
    fig = Figure(figsize=(4.8, 4.2))
    ax = fig.subplots()
    edges = np.arange(6) + 0.5
    mesh = ax.pcolormesh(edges, edges, np.ma.masked_invalid(share), cmap=cmap, vmin=0, vmax=1,
                         edgecolors="#fcfcfb", linewidth=2)  # fmt: skip
    for i in range(5):
        for j in range(5):
            if totals[i, 0] == 0:
                continue
            dark = share[i, j] > 0.55
            ax.text(j + 1, i + 1, str(counts[i, j]), ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK)  # fmt: skip
    for i in np.flatnonzero(totals[:, 0] == 0):
        ax.text(3, i + 1, "no windows with this grade", ha="center", va="center", fontsize=8,
                color=MUTED)  # fmt: skip
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(5.5, 0.5)  # grade 1 at the top
    ax.set_aspect("equal")
    ax.set_xlabel("predicted grade")
    ax.set_ylabel("my grade (latest normal-mode label)")
    _style(ax)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("share of the row (my grade)", color=INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_2, length=0)
    ax.set_title(("SYNTHETIC: " if prov.synthetic else "") + f"Grade confusion, {model}, test "
                 "sessions", color=INK, fontsize=10)  # fmt: skip
    return _save(fig, out_stem, prov.synthetic, prov.dirty)


def reliability_chart(
    calibration_json: str | Path, out_stem: str | Path, models: Sequence[str] | None = None
) -> list[Path]:
    """Reliability diagram: accuracy vs mean confidence per bin, windows per bin underneath."""
    from matplotlib.figure import Figure

    payload = json.loads(Path(calibration_json).read_text())
    rows, prov = reliability_rows(payload, models)
    _check_out(Path(out_stem), prov.synthetic, prov.dirty)
    met = payload["metrics"]
    n_bins = int(met.get("ece_bins", 10))
    fig = Figure(figsize=(5.2, 5.2))
    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    top.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, label="perfectly calibrated")
    width = 1.0 / n_bins / (len(rows) + 1)
    from gaitlab.eval.run_eval import LADDER_ORDER

    for k, (name, (conf, acc, count)) in enumerate(rows.items()):
        color = series_color(name)
        rung = LADDER_ORDER.index(name) if name in LADDER_ORDER else len(MARKERS) - 1
        label = f"{name} (ECE {float(met['models'][name]['ece']):.2f})"
        top.plot(conf, acc, lw=2, color=color, marker=MARKERS[rung], ms=7,
                 markeredgecolor="#fcfcfb", markeredgewidth=1, label=label)  # fmt: skip
        centers = (np.floor(conf * n_bins).clip(0, n_bins - 1) + 0.5) / n_bins
        offset = (k - (len(rows) - 1) / 2) * width
        bottom.bar(centers + offset, count, width=width * 0.9, color=color, lw=0)
    top.set_xlim(-0.02, 1.02)  # room for markers at 0 and 1
    top.set_ylim(-0.03, 1.03)
    top.set_ylabel("accuracy (predicted grade = my grade)")
    top.grid(color=GRID, lw=0.5)
    top.set_axisbelow(True)
    top.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)
    bottom.set_xlabel(f"confidence (max P over grades 1-5), {n_bins} equal-width bins")
    bottom.set_ylabel("windows")
    bottom.yaxis.get_major_locator().set_params(integer=True)
    for ax in (top, bottom):
        _style(ax)
    top.set_title(("SYNTHETIC: " if prov.synthetic else "") + "Grade calibration, test sessions",
                  color=INK, fontsize=10)  # fmt: skip
    return _save(fig, out_stem, prov.synthetic, prov.dirty)


def bland_altman_chart(
    agreement_json: str | Path,
    pairs_parquet: str | Path,
    out_stem: str | Path,
    features: Sequence[str] | None = None,
) -> list[Path]:
    """V7 small multiples: per feature, a − b vs (a + b) / 2 with bias and limits of agreement."""
    import pandas as pd
    from matplotlib.figure import Figure

    payload = json.loads(Path(agreement_json).read_text())
    want = payload["metrics"].get("pairs_sha256")
    got = hashlib.sha256(Path(pairs_parquet).read_bytes()).hexdigest()
    if want != got:
        raise ValueError(f"{pairs_parquet} sha256 does not match the result's pairs_sha256: "
                         "rerun `gaitlab validate own` so both come from one run")  # fmt: skip
    rows, prov = bland_altman_rows(payload, pd.read_parquet(pairs_parquet), features)
    _check_out(Path(out_stem), prov.synthetic, prov.dirty)
    a_name, b_name = payload["metrics"]["backends"]
    ncols = min(len(rows), 5)
    nrows = -(-len(rows) // ncols)
    fig = Figure(figsize=(2.6 * ncols, 2.4 * nrows + 0.4), layout="constrained")
    axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    for ax, (f, (mean, diff, s)) in zip(axes, rows.items(), strict=False):
        ax.scatter(mean, diff, s=8, color=SERIES_LIGHT[0], alpha=0.5, lw=0)
        if s.get("bias") is not None:
            ax.axhline(s["bias"], color=INK_2, lw=1)
        for k in ("loa_low", "loa_high"):
            if s.get(k) is not None:
                ax.axhline(s[k], color=MUTED, lw=1, ls="--")
        ax.axhline(0, color=AXIS, lw=0.5)
        ax.set_title(f"{f} (n={s.get('n', len(diff))})", fontsize=8, color=INK)
        ax.tick_params(labelsize=7)
        _style(ax)
    for ax in axes[len(rows) :]:
        ax.set_visible(False)
    fig.supxlabel(f"mean of {a_name} and {b_name}, per matched stride", fontsize=9, color=INK_2)
    fig.supylabel(f"{a_name} − {b_name}", fontsize=9, color=INK_2)
    title = f"Pose backend agreement, {a_name} vs {b_name} (solid: bias, dashed: 95% limits)"
    fig.suptitle(("SYNTHETIC: " if prov.synthetic else "") + title, fontsize=10, color=INK)
    return _save(fig, out_stem, prov.synthetic, prov.dirty)
