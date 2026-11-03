"""`gaitlab` CLI (TECHNICAL_PLAN §10). Thin orchestration; logic lives in the modules."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

from gaitlab.config import load_features_config, load_pipeline_config

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


@app.command()
def process(
    session: str = typer.Option(...), backend: str = "mediapipe", config: str = CFG
) -> None:
    """KeypointSeq → strides.parquet + windows.parquet for a session."""
    import pandas as pd

    from gaitlab.data.sessions import load_sessions
    from gaitlab.gait.features import analyze_clip
    from gaitlab.gait.windows import make_windows
    from gaitlab.types import KeypointSeq

    cfg = load_pipeline_config(config)
    rows = [r for r in load_sessions() if r.session_id == session]
    if not rows:
        raise typer.BadParameter(f"no rows for session {session} in data/sessions.csv")
    interim = Path("data/interim") / session
    standing = next((r for r in rows if r.condition == "standing"), None)
    standing_seq = (
        KeypointSeq.load(str(interim / f"{standing.clip_id}.{backend}.npz")) if standing else None
    )
    strides_out, windows_out = [], []
    for r in rows:
        if r.condition == "standing":
            continue
        seq = KeypointSeq.load(str(interim / f"{r.clip_id}.{backend}.npz"))
        res = analyze_clip(seq, cfg, standing=standing_seq)
        for s in res.strides:
            strides_out.append({"session_id": session, "clip_id": r.clip_id, **s.__dict__})
        wins = make_windows(res.strides, session, r.clip_id, cfg.windows, r.condition, r.speed_kmh)
        windows_out.extend(w.flat() for w in wins)
    out = Path("data/processed") / session
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(strides_out).to_parquet(out / "strides.parquet", index=False)
    pd.DataFrame(windows_out).to_parquet(out / "windows.parquet", index=False)
    typer.echo(f"{len(strides_out)} strides, {len(windows_out)} windows → {out}")


@app.command()
def synth(
    n_sessions: int = typer.Option(6),
    seed: int = 0,
    out: str = "data/synthetic",
    config: str = CFG,
    features: str = FCFG,
) -> None:
    """Synthetic sessions + rule labels for dry runs. Everything is marked synthetic."""
    import pandas as pd

    from gaitlab.data.synthetic import synthetic_features_config, synthetic_sessions

    cfg = load_pipeline_config(config)
    fcfg = synthetic_features_config(load_features_config(features))
    windows, labels, sessions = synthetic_sessions(n_sessions, cfg, fcfg, seed=seed)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(windows).to_parquet(o / "windows.parquet", index=False)
    with open(o / "labels.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(labels[0].keys()))
        wr.writeheader()
        wr.writerows(labels)
    (o / "sessions.json").write_text(json.dumps(sessions, indent=2))
    (o / "SYNTHETIC").write_text("Everything in this folder is synthetic. Never report it.\n")
    typer.echo(f"{len(windows)} synthetic windows from {n_sessions} sessions → {o}")


@app.command()
def overlay(clip: str = typer.Option(...), gif: bool = False) -> None:
    """Skeleton + contacts + angles video (TASKS M0)."""
    _todo("overlay (viz.overlay)")


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
