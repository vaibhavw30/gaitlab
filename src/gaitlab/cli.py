"""`gaitlab` CLI (TECHNICAL_PLAN §10). Thin orchestration; logic lives in the modules."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from gaitlab.config import load_features_config, load_pipeline_config

if TYPE_CHECKING:
    from gaitlab.config import PipelineConfig
    from gaitlab.types import KeypointSeq

app = typer.Typer(no_args_is_help=True, add_completion=False, help="GaitLab running-form grader")

CFG = typer.Option("configs/default.yaml", help="pipeline config")
FCFG = typer.Option("configs/features.yaml", help="features/tags config")


def _todo(task: str) -> None:
    typer.secho(f"Not implemented yet: {task} (see docs/TASKS.md)", fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


@app.command()
def extract(
    session: str = typer.Option(...),
    backend: str = typer.Option("mediapipe"),
    allow_low_fps: bool = False,
    config: str = CFG,
) -> None:
    """video → KeypointSeq npz for every clip in a session (TASKS M0)."""
    _todo("extract (io.video + pose backends)")


def _clip_ids(clip: Path) -> tuple[str, str]:
    """``<dir>/<session>/<clip>.<backend>.npz`` → (session id, clip id)."""
    return clip.parent.name, clip.name.split(".")[0]


def _condition_from_clip_id(clip_id: str) -> str:
    """``natural`` / ``natural-2`` → ``natural``; anything else → ``""``."""
    from typing import get_args

    from gaitlab.data.sessions import Condition

    cond = clip_id.split("-")[0]
    return cond if cond in get_args(Condition) else ""


def _analyze_rows(
    seq: KeypointSeq,
    cfg: PipelineConfig,
    session_id: str,
    clip_id: str,
    condition: str,
    speed_kmh: float | None,
    standing_seq: KeypointSeq | None,
) -> tuple[list[dict], list[dict]]:
    """One clip → (strides rows, windows rows), each tagged with the sequence's synthetic flag."""
    from gaitlab.gait.features import analyze_clip
    from gaitlab.gait.windows import make_windows

    res = analyze_clip(seq, cfg, standing=standing_seq)
    flag = {"synthetic": seq.synthetic}
    strides = [
        {"session_id": session_id, "clip_id": clip_id, **s.__dict__, **flag} for s in res.strides
    ]
    wins = make_windows(res.strides, session_id, clip_id, cfg.windows, condition, speed_kmh)
    return strides, [{**w.flat(), **flag} for w in wins]


def _write_processed(
    out: Path, prefix: str, strides: list[dict], windows: list[dict], synthetic: bool
) -> None:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(strides).to_parquet(out / f"{prefix}strides.parquet", index=False)
    pd.DataFrame(windows).to_parquet(out / f"{prefix}windows.parquet", index=False)
    if synthetic:
        (out / "SYNTHETIC").write_text(SYNTHETIC_MARKER)
    typer.echo(
        f"{len(strides)} strides, {len(windows)} windows → {out}"
        + (" (synthetic)" if synthetic else "")
    )


SYNTHETIC_MARKER = "Everything in this folder is synthetic. Never report it.\n"


@app.command()
def process(
    session: str | None = typer.Option(None, help="session id: every clip in its sessions.csv"),
    clip: str | None = typer.Option(None, help="one KeypointSeq npz; no sessions row needed"),
    sessions: str = typer.Option(
        "data/sessions.csv",
        help="sessions CSV for --session; npz are read from <its dir>/interim/<session>/",
    ),
    backend: str = typer.Option("mediapipe", help="npz backend suffix for --session"),
    session_id: str | None = typer.Option(None, help="--clip: default is the npz's folder name"),
    condition: str | None = typer.Option(None, help="--clip: default inferred from the clip id"),
    speed_kmh: float | None = typer.Option(None, help="--clip: treadmill speed, km/h"),
    standing: str | None = typer.Option(None, help="--clip: standing-clip npz for foot angle"),
    out: str | None = typer.Option(None, help="output folder (default: see TECHNICAL_PLAN §10)"),
    config: str = CFG,
) -> None:
    """KeypointSeq → strides + windows parquet, for a session or a single clip.

    Outputs from synthetic keypoints carry ``synthetic=True`` and a SYNTHETIC marker file.
    """
    from gaitlab.types import KeypointSeq

    if (session is None) == (clip is None):
        raise typer.BadParameter("give exactly one of --session or --clip")
    cfg = load_pipeline_config(config)

    if clip is not None:
        path = Path(clip)
        sid, cid = _clip_ids(path)
        sid = session_id or sid
        seq = KeypointSeq.load(str(path))
        standing_seq = KeypointSeq.load(standing) if standing else None
        cond = condition if condition is not None else _condition_from_clip_id(cid)
        strides, windows = _analyze_rows(seq, cfg, sid, cid, cond, speed_kmh, standing_seq)
        root = Path("data/synthetic") if seq.synthetic else Path("data")
        dest = Path(out) if out else root / "processed" / sid
        _write_processed(dest, f"{cid}.", strides, windows, seq.synthetic)
        return

    from gaitlab.data.sessions import load_sessions

    rows = [r for r in load_sessions(sessions) if r.session_id == session]
    if not rows:
        raise typer.BadParameter(f"no rows for session {session} in {sessions}")
    base = Path(sessions).parent
    interim = base / "interim" / session
    standing_row = next((r for r in rows if r.condition == "standing"), None)
    standing_seq = (
        KeypointSeq.load(str(interim / f"{standing_row.clip_id}.{backend}.npz"))
        if standing_row
        else None
    )
    strides_out, windows_out, flags = [], [], set()
    for r in rows:
        if r.condition == "standing":
            continue
        seq = KeypointSeq.load(str(interim / f"{r.clip_id}.{backend}.npz"))
        flags.add(seq.synthetic)
        s_rows, w_rows = _analyze_rows(
            seq, cfg, session, r.clip_id, r.condition, r.speed_kmh, standing_seq
        )
        strides_out.extend(s_rows)
        windows_out.extend(w_rows)
    if len(flags) > 1:
        raise typer.BadParameter(f"session {session} mixes synthetic and real keypoints")
    synthetic = flags == {True}
    dest = Path(out) if out else base / "processed" / session
    _write_processed(dest, "", strides_out, windows_out, synthetic)


@app.command()
def synth(
    n_sessions: int = typer.Option(6),
    seed: int = 0,
    out: str = "data/synthetic",
    write_keypoints: bool = typer.Option(
        False,
        help="also save each clip as <out>/interim/<session>/<clip>.synthetic.npz "
        "plus <out>/sessions.csv",
    ),
    config: str = CFG,
    features: str = FCFG,
) -> None:
    """Synthetic sessions + rule labels for dry runs. Everything is marked synthetic."""
    import pandas as pd

    from gaitlab.data.synthetic import (
        save_synthetic_keypoints,
        synthetic_features_config,
        synthetic_session_clips,
        synthetic_sessions,
    )

    cfg = load_pipeline_config(config)
    fcfg = synthetic_features_config(load_features_config(features))
    clips = synthetic_session_clips(n_sessions, seed=seed)
    windows, labels, sessions = synthetic_sessions(n_sessions, cfg, fcfg, seed=seed, clips=clips)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "SYNTHETIC").write_text(SYNTHETIC_MARKER)
    pd.DataFrame([{**w, "synthetic": True} for w in windows]).to_parquet(
        o / "windows.parquet", index=False
    )
    with open(o / "labels.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(labels[0].keys()))
        wr.writeheader()
        wr.writerows(labels)
    (o / "sessions.json").write_text(json.dumps(sessions, indent=2))
    typer.echo(f"{len(windows)} synthetic windows from {n_sessions} sessions → {o}")
    if write_keypoints:
        csv_path = save_synthetic_keypoints(clips, o)
        typer.echo(f"{len(clips)} synthetic clips → {o / 'interim'}, sessions → {csv_path}")
        typer.echo(
            f"next: gaitlab process --session {clips[0].session_id} --sessions {csv_path} "
            "--backend synthetic"
        )


@app.command()
def overlay(
    clip: str = typer.Option(..., help="KeypointSeq npz"),
    video: str | None = typer.Option(None, help="source video; plain canvas if omitted"),
    gif: bool = typer.Option(False, help="write a GIF (480 px wide, 15 fps) instead of mp4"),
    out: str | None = typer.Option(None, help="default: <clip without .npz>.overlay.{mp4,gif}"),
    max_seconds: float | None = typer.Option(None, help="only the first N seconds"),
    config: str = CFG,
) -> None:
    """Skeleton + IC/TO markers + near-side knee and shin angles, on the video or a canvas."""
    from gaitlab.gait.features import analyze_clip
    from gaitlab.types import KeypointSeq
    from gaitlab.viz.overlay import render_overlay

    seq = KeypointSeq.load(clip)
    analysis = analyze_clip(seq, load_pipeline_config(config))
    stem = clip[: -len(".npz")] if clip.endswith(".npz") else clip
    dest = out or f"{stem}.overlay.{'gif' if gif else 'mp4'}"
    path = render_overlay(seq, analysis, dest, video_path=video, gif=gif, max_seconds=max_seconds)
    typer.echo(f"overlay → {path}" + (" (synthetic)" if seq.synthetic else ""))


@app.command()
def label(blind: int = typer.Option(0, help="blind re-grade N windows")) -> None:
    """Labeling app (TASKS M2)."""
    _todo("labeling app (label.app)")


@app.command()
def baselines() -> None:
    """Rules / logistic regression / LightGBM (TASKS M2)."""
    _todo("baselines (models.baselines)")


@app.command("build-sft")
def build_sft() -> None:
    """labels + windows → data/sft/*.jsonl by session split (TASKS M3)."""
    _todo("build-sft (llm.sft_data + data.splits)")


@app.command()
def train(model: str = typer.Option("1b"), allow_synthetic: bool = False) -> None:
    """LoRA fine-tune (TASKS M3)."""
    _todo("train (llm.train_lora)")


@app.command()
def predict(model: str = typer.Option("1b")) -> None:
    """Grade + tags + cue on the test split (TASKS M3)."""
    _todo("predict (llm.grade)")


@app.command("eval")
def eval_() -> None:
    """All grader metrics → results/grader (TASKS M3)."""
    _todo("eval (eval.run_eval)")


@app.command()
def validate(target: str = typer.Argument(..., help="biocv | mocap")) -> None:
    """Event/feature validation on public data (TASKS M0)."""
    _todo(f"validate {target}")


@app.command()
def bench(trt: bool = False) -> None:
    """Latency and TensorRT benchmarks (TASKS M4)."""
    _todo("bench (bench.latency, bench.trt)")


@app.command()
def report(readme: str = "README.md", root: str = "results") -> None:
    """Regenerate the README results block from results/*.json."""
    from gaitlab.report.readme_table import update_readme

    changed = update_readme(readme, root)
    typer.echo("README updated" if changed else "README unchanged")


@app.command("app")
def app_() -> None:
    """Coach demo app (TASKS M4)."""
    _todo("coach app (app.coach)")


if __name__ == "__main__":
    app()
