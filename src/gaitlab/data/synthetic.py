"""Synthetic side-view treadmill runner with exact ground truth. Test fixture and dry-run data.

Everything produced here is synthetic: results computed from it must set ``synthetic=True``
(CLAUDE.md rule 3). The model is kinematic, not physical. It exists so events and features can
be checked against values known by construction, not to look like a real runner.

Model (sagittal plane, metres, x = forward along the facing direction, y = up, hip at x = 0):
- Each foot cycles with period T = 120 / cadence. Stance is the first ``duty_factor`` of the
  cycle; the far foot is offset by half a cycle.
- Stance: the ground-contact point moves backward at belt speed. Heel-first for the first 30% of
  stance (toe lowering from ``foot_strike_deg``), flat until 70%, then toe-pivot (heel rising to
  ``toe_off_deg``). min(heel_y, toe_y) == 0 throughout stance.
- Swing: heel x eases (cosine) from its toe-off position back to ``overstride_m``; the foot angle
  goes linearly from -toe_off to +strike; the lowest foot point follows
  ``clearance * sqrt(sin(pi*s))``.
- Hip height oscillates at 2/T with minima at each mid-stance (peak-to-peak ``vert_osc_m``).
- Knee from 2-link inverse kinematics, bending forward. Shoulder from trunk lean.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from gaitlab.pose.skeleton import CANONICAL, J
from gaitlab.types import KeypointSeq


@dataclass(frozen=True)
class GaitParams:
    cadence_spm: float = 168.0
    duty_factor: float = 0.36
    speed_mps: float = 3.3
    overstride_m: float = 0.25
    foot_strike_deg: float = 10.0
    toe_off_deg: float = 30.0
    vert_osc_m: float = 0.08
    trunk_lean_deg: float = 6.0
    hip_height_m: float = 0.97
    thigh_m: float = 0.5
    shank_m: float = 0.5
    foot_m: float = 0.25
    trunk_m: float = 0.55
    swing_clearance_m: float = 0.25
    direction: int = 1
    near_side: str = "r"
    t0_s: float = 0.1

    @property
    def stride_time_s(self) -> float:
        return 120.0 / self.cadence_spm

    @property
    def leg_m(self) -> float:
        return self.thigh_m + self.shank_m


@dataclass(frozen=True)
class NoiseParams:
    jitter_px: float = 0.0
    dropout_frac: float = 0.0
    near_vis: float = 0.95
    far_vis: float = 0.7
    seed: int = 0


@dataclass
class SyntheticTruth:
    params: GaitParams
    fps: float
    px_per_m: float
    ic: dict[str, np.ndarray]
    to: dict[str, np.ndarray]
    xy_clean: np.ndarray  # (T, J, 2) y-down pixels, noise-free
    stance: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def leg_length_px(self) -> float:
        return self.params.leg_m * self.px_per_m

    @property
    def overstride_L(self) -> float:
        return self.params.overstride_m / self.params.leg_m

    @property
    def vert_osc_L(self) -> float:
        return self.params.vert_osc_m / self.params.leg_m


# Conditions used by `gaitlab synth` and tests: deltas from natural running.
CONDITIONS: dict[str, dict[str, float]] = {
    "natural": {},
    "overstride": {"overstride_m": 0.40, "foot_strike_deg": 18.0, "cadence_spm": 158.0},
    "metronome": {"cadence_spm": 182.0, "overstride_m": 0.15, "vert_osc_m": 0.06},
    "lean": {"trunk_lean_deg": 16.0},
    "fatigue": {"vert_osc_m": 0.11, "cadence_spm": 162.0, "overstride_m": 0.30},
}


def condition_params(base: GaitParams, condition: str) -> GaitParams:
    return replace(base, **CONDITIONS[condition])  # type: ignore[arg-type]


def _foot_kinematics(
    t: np.ndarray, p: GaitParams, offset: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heel (T,2), toe (T,2) in metres (x forward, y up) and a stance bool (T,)."""
    T = p.stride_time_s
    df = p.duty_factor
    lf = p.foot_m
    alpha = np.radians(p.foot_strike_deg)
    beta = np.radians(p.toe_off_deg)
    belt = p.speed_mps * df * T  # backward travel of the contact point during stance

    phi = np.mod((t - p.t0_s) / T + offset, 1.0)
    stance = phi < df
    heel = np.zeros((len(t), 2))
    toe = np.zeros((len(t), 2))

    u = np.where(stance, phi / df, 0.0)
    travel = belt * u
    # Phase A: heel on ground, toe lowering.
    a = stance & (u < 0.3)
    th = alpha * (1 - u / 0.3)
    heel[a, 0] = p.overstride_m - travel[a]
    toe[a, 0] = heel[a, 0] + lf * np.cos(th[a])
    toe[a, 1] = lf * np.sin(th[a])
    # Phase B: flat.
    b = stance & (u >= 0.3) & (u < 0.7)
    heel[b, 0] = p.overstride_m - travel[b]
    toe[b, 0] = heel[b, 0] + lf
    # Phase C: toe on ground, heel rising.
    c = stance & (u >= 0.7)
    th_c = -beta * (u - 0.7) / 0.3
    toe[c, 0] = p.overstride_m + lf - travel[c]
    heel[c, 0] = toe[c, 0] - lf * np.cos(th_c[c])
    heel[c, 1] = -lf * np.sin(th_c[c])

    # Swing.
    sw = ~stance
    s = np.where(sw, (phi - df) / (1 - df), 0.0)
    toe_x_end = p.overstride_m + lf - belt
    heel_x_end = toe_x_end - lf * np.cos(-beta)
    hx = heel_x_end + (p.overstride_m - heel_x_end) * (1 - np.cos(np.pi * s)) / 2
    th_s = -beta + (alpha + beta) * s
    h = p.swing_clearance_m * np.sqrt(np.clip(np.sin(np.pi * s), 0.0, None))
    lift = h - np.minimum(0.0, lf * np.sin(th_s))
    heel[sw, 0] = hx[sw]
    heel[sw, 1] = lift[sw]
    toe[sw, 0] = hx[sw] + lf * np.cos(th_s[sw])
    toe[sw, 1] = lift[sw] + lf * np.sin(th_s[sw])
    return heel, toe, stance


def _ankle_from_foot(heel: np.ndarray, toe: np.ndarray) -> np.ndarray:
    along = toe - heel
    along /= np.linalg.norm(along, axis=-1, keepdims=True)
    perp = np.stack([-along[:, 1], along[:, 0]], axis=-1)
    return heel + 0.06 * along + 0.08 * perp


def _knee_ik(hip: np.ndarray, ankle: np.ndarray, l1: float, l2: float) -> np.ndarray:
    v = ankle - hip
    r = np.linalg.norm(v, axis=-1)
    r_c = np.clip(r, abs(l1 - l2) + 1e-6, 0.999 * (l1 + l2))
    u = v / r[:, None]
    cos_g = (l1**2 + r_c**2 - l2**2) / (2 * l1 * r_c)
    g = np.arccos(np.clip(cos_g, -1, 1))
    n = np.stack([-u[:, 1], u[:, 0]], axis=-1)  # forward-pointing normal for a downward leg
    return hip + l1 * (np.cos(g)[:, None] * u + np.sin(g)[:, None] * n)


def _runs_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    keep = (starts > 0) & (ends < len(mask) - 1)
    return starts[keep], ends[keep]


def synth_clip(
    params: GaitParams | None = None,
    fps: float = 120.0,
    duration_s: float = 12.0,
    noise: NoiseParams | None = None,
    size: tuple[int, int] = (1920, 1080),
    px_per_m: float = 500.0,
    standing: bool = False,
) -> tuple[KeypointSeq, SyntheticTruth]:
    """Generate a clip and its ground truth.

    With ``standing=True`` the runner stands still with flat feet (foot-angle calibration clip).
    The default scale fits the whole body (head about 1.8 m above the belt) in a 1080-px-high
    frame with the belt 100 px above the bottom edge.
    """
    p = params or GaitParams()
    nz = noise or NoiseParams()
    n = round(duration_s * fps)
    t = np.arange(n) / fps
    far = "l" if p.near_side == "r" else "r"

    hip = np.zeros((n, 2))
    feet: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if standing:
        hip[:, 1] = p.hip_height_m + 0.02
        for side in (p.near_side, far):
            heel = np.zeros((n, 2))
            toe = np.zeros((n, 2))
            heel[:, 0] = -0.05
            toe[:, 0] = -0.05 + p.foot_m
            feet[side] = (heel, toe, np.ones(n, dtype=bool))
    else:
        T = p.stride_time_s
        t_ms = p.t0_s + p.duty_factor * T / 2
        hip[:, 1] = p.hip_height_m - (p.vert_osc_m / 2) * np.cos(2 * np.pi * (t - t_ms) / (T / 2))
        feet[p.near_side] = _foot_kinematics(t, p, 0.0)
        feet[far] = _foot_kinematics(t, p, 0.5)

    lean = np.radians(0.0 if standing else p.trunk_lean_deg)
    shoulder = hip + p.trunk_m * np.array([np.sin(lean), np.cos(lean)])
    nose = shoulder + np.array([0.08, 0.22])

    pts = np.zeros((n, len(CANONICAL), 2))
    pts[:, J["nose"]] = nose
    for side, (heel, toe, _) in feet.items():
        ankle = _ankle_from_foot(heel, toe)
        knee = _knee_ik(hip, ankle, p.thigh_m, p.shank_m)
        pts[:, J[f"{side}_shoulder"]] = shoulder
        pts[:, J[f"{side}_hip"]] = hip
        pts[:, J[f"{side}_knee"]] = knee
        pts[:, J[f"{side}_ankle"]] = ankle
        pts[:, J[f"{side}_heel"]] = heel
        pts[:, J[f"{side}_toe"]] = toe

    w, h = size
    ground_px = 100.0
    xy = np.empty_like(pts)
    xy[..., 0] = w / 2 + p.direction * pts[..., 0] * px_per_m
    xy[..., 1] = h - (ground_px + pts[..., 1] * px_per_m)  # y-down

    vis = np.full((n, len(CANONICAL)), nz.near_vis, dtype=np.float32)
    for name in CANONICAL:
        if name.startswith(f"{far}_"):
            vis[:, J[name]] = nz.far_vis

    rng = np.random.default_rng(nz.seed)
    noisy = xy.copy()
    if nz.jitter_px > 0:
        noisy += rng.normal(0.0, nz.jitter_px, size=noisy.shape)
    if nz.dropout_frac > 0:
        drop = rng.random(vis.shape) < nz.dropout_frac
        vis = vis.copy()
        vis[drop] = 0.1

    ic: dict[str, np.ndarray] = {}
    to: dict[str, np.ndarray] = {}
    stance_masks: dict[str, np.ndarray] = {}
    for side, (_, _, stance) in feet.items():
        stance_masks[side] = stance
        if not standing:
            ic[side], to[side] = _runs_bounds(stance)

    seq = KeypointSeq(
        xy=noisy.astype(np.float32),
        vis=vis,
        t=t,
        fps=fps,
        size=size,
        backend="synthetic",
        meta={"synthetic": True, "params": p.__dict__, "noise": nz.__dict__},
    )
    truth = SyntheticTruth(
        params=p,
        fps=fps,
        px_per_m=px_per_m,
        ic=ic,
        to=to,
        xy_clean=xy,
        stance=stance_masks,
    )
    return seq, truth


# ---------------------------------------------------------------------------------------------
# Synthetic sessions for dry runs (`gaitlab synth`). Rule-derived labels; synthetic=True always.
# ---------------------------------------------------------------------------------------------

# Thresholds that separate the synthetic conditions above. Valid ONLY for synthetic data; real
# thresholds are calibrated in M2 (ADR-0009) and live in configs/features.yaml.
SYNTHETIC_THRESHOLDS: dict[str, float] = {
    "overstride": 0.32,
    "low_cadence": 163.0,
    "high_bounce": 0.095,
    "stiff_knee_landing": 35.0,
    "steep_heel_strike": 13.0,
    "forward_lean": 11.0,
    "backward_lean": -5.0,
}

CUE_TEMPLATES: dict[str, str] = {
    "overstride": "Land closer under your hips instead of reaching the foot out front.",
    "low_cadence": "Take quicker steps to raise your cadence a little.",
    "high_bounce": "Run lower with less bounce; push forward, not up.",
    "stiff_knee_landing": "Land with a soft, slightly bent knee.",
    "steep_heel_strike": "Land flatter rather than on a steep heel with the toe up.",
    "forward_lean": "Run tall; lean from the ankles, not the waist, for better posture.",
    "backward_lean": "Stop leaning back; stay upright over your hips.",
}


def synthetic_features_config(base):
    """Copy of a FeaturesConfig with SYNTHETIC_THRESHOLDS filled in (synthetic use only)."""
    cfg = base.model_copy(deep=True)
    for tag, rule in cfg.tags.items():
        rule.threshold = SYNTHETIC_THRESHOLDS[tag]
    return cfg


@dataclass
class SyntheticClip:
    """One generated clip of a synthetic session (``gaitlab synth``)."""

    session_id: str
    date: str  # ISO date
    clip_id: str
    condition: str
    speed_kmh: float
    seq: KeypointSeq


SYNTHETIC_SPEED_KMH = 12.0
"""Speed written to synthetic windows and session rows (km/h)."""


def synthetic_session_clips(
    n_sessions: int,
    seed: int = 0,
    clip_seconds: float = 32.0,
    fps: float = 120.0,
) -> list[SyntheticClip]:
    """Generate sessions × conditions as keypoint clips (noise on), all marked synthetic.

    Each session perturbs the base gait slightly and picks a random facing direction. Session ids
    are weekly dates from 2026-01-01 (``YYYYMMDD-01``); clip id = condition.
    """
    import datetime as dt

    rng = np.random.default_rng(seed)
    clips: list[SyntheticClip] = []
    for s in range(n_sessions):
        day = dt.date(2026, 1, 1) + dt.timedelta(days=7 * s)
        sid = f"{day:%Y%m%d}-01"
        base = GaitParams(
            cadence_spm=168.0 + rng.normal(0, 3),
            overstride_m=0.25 + rng.normal(0, 0.02),
            trunk_lean_deg=6.0 + rng.normal(0, 1.5),
            vert_osc_m=0.08 + rng.normal(0, 0.006),
            direction=int(rng.choice([-1, 1])),
        )
        for cond in CONDITIONS:
            seq, _ = synth_clip(
                condition_params(base, cond),
                fps=fps,
                duration_s=clip_seconds,
                noise=NoiseParams(jitter_px=3.0, dropout_frac=0.01, seed=int(rng.integers(1e9))),
            )
            seq.meta.update({"session_id": sid, "clip_id": cond, "condition": cond})
            clips.append(SyntheticClip(sid, day.isoformat(), cond, cond, SYNTHETIC_SPEED_KMH, seq))
    return clips


def synthetic_sessions(
    n_sessions: int,
    pipeline_cfg,
    features_cfg,
    seed: int = 0,
    clip_seconds: float = 32.0,
    fps: float = 120.0,
    clips: list[SyntheticClip] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Analyze synthetic clips → windows rows, rule labels and session rows.

    Clips come from ``synthetic_session_clips(n_sessions, seed, clip_seconds, fps)`` unless
    passed in. Returns (windows, labels, sessions), all marked synthetic. ``features_cfg`` must
    already have thresholds (use ``synthetic_features_config``).
    """
    from gaitlab.gait.features import analyze_clip
    from gaitlab.gait.windows import make_windows
    from gaitlab.models.rules import rules_predict

    if clips is None:
        clips = synthetic_session_clips(n_sessions, seed, clip_seconds, fps)
    windows: list[dict] = []
    labels: list[dict] = []
    sessions: list[dict] = []
    for c in clips:
        analysis = analyze_clip(c.seq, pipeline_cfg)
        wins = make_windows(
            analysis.strides, c.session_id, c.clip_id, pipeline_cfg.windows, c.condition,
            c.speed_kmh,
        )  # fmt: skip
        for w in wins:
            row = w.flat()
            grade, tags = rules_predict(row, features_cfg)  # type: ignore[arg-type]
            cue = CUE_TEMPLATES[tags[0]] if tags else ""
            windows.append(row)
            labels.append(
                {
                    "window_id": w.window_id,
                    "grade": grade,
                    "tags": ";".join(tags),
                    "cue": cue,
                    "rater": "synthetic",
                    "mode": "normal",
                    "labeled_at": f"{c.date}T12:00:00",
                }
            )
        sessions.append(
            {"session_id": c.session_id, "date": c.date, "clip_id": c.clip_id, "synthetic": True}
        )
    return windows, labels, sessions


def load_synthetic_grader_inputs(
    root: str | Path = "data/synthetic",
) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    """``gaitlab synth`` output → (window rows, window_id → label, session split).

    The split is by session date (``data.splits.make_session_split``) over ``sessions.json`` and
    is written once to ``<root>/splits.json`` (frozen like ``data/splits.json``, but separate).
    """
    import json

    import pandas as pd

    from gaitlab.data.sessions import latest_normal_labels, load_labels
    from gaitlab.data.splits import make_session_split, write_split_frozen

    root = Path(root)
    if not (root / "SYNTHETIC").exists():
        raise FileNotFoundError(f"{root} has no SYNTHETIC marker; run gaitlab synth first")
    windows = pd.read_parquet(root / "windows.parquet").to_dict("records")
    if not all(w.get("synthetic") is True for w in windows):
        raise ValueError(f"{root}/windows.parquet has rows not flagged synthetic")
    labels = {
        wid: {"grade": r.grade, "tags": r.tags, "cue": r.cue}
        for wid, r in latest_normal_labels(load_labels(root / "labels.csv")).items()
    }
    sessions = json.loads((root / "sessions.json").read_text())
    split = make_session_split([(s["session_id"], s["date"]) for s in sessions])
    write_split_frozen(split, root / "splits.json")
    return windows, labels, split


def save_synthetic_keypoints(clips: list[SyntheticClip], root: str | Path) -> Path:
    """Write each clip as ``<root>/interim/<session>/<clip>.synthetic.npz`` plus a sessions CSV.

    ``<root>/sessions.csv`` follows the ``data/sessions.csv`` schema (TECHNICAL_PLAN §4.6) with
    ``path`` pointing at the npz, and one extra column ``synthetic`` (always true), so
    ``gaitlab process --session S --sessions <root>/sessions.csv --backend synthetic`` runs on it.
    Returns the CSV path.
    """
    import csv

    from gaitlab.data.sessions import SessionRow

    root = Path(root)
    rows = []
    for c in clips:
        if not c.seq.synthetic:
            raise ValueError(f"{c.session_id}/{c.clip_id}: not marked synthetic")
        path = root / "interim" / c.session_id / f"{c.clip_id}.synthetic.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        c.seq.save(str(path))
        row = SessionRow(
            session_id=c.session_id,
            date=c.date,
            clip_id=c.clip_id,
            path=str(path),
            condition=c.condition,
            speed_kmh=c.speed_kmh,
            fps=c.seq.fps,
            notes="synthetic keypoints (gaitlab synth --write-keypoints); never report",
        )
        rows.append({**row.model_dump(mode="json"), "synthetic": True})
    out = root / "sessions.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=[*SessionRow.model_fields, "synthetic"])
        wr.writeheader()
        wr.writerows([{k: ("" if v is None else v) for k, v in r.items()} for r in rows])
    return out
