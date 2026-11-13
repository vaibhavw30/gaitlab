"""`gaitlab` CLI (TECHNICAL_PLAN §10). Thin orchestration; logic lives in the modules."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import typer

from gaitlab.config import load_features_config, load_pipeline_config

if TYPE_CHECKING:
    from gaitlab.config import PipelineConfig

app = typer.Typer(no_args_is_help=True, add_completion=False, help="GaitLab running-form grader")

CFG = typer.Option("configs/default.yaml", help="pipeline config")
FCFG = typer.Option("configs/features.yaml", help="features/tags config")
ONLY_SESSIONS = typer.Option(None, help="only these sessions (repeat the option)")


def _todo(task: str) -> None:
    typer.secho(f"Not implemented yet: {task} (see docs/TASKS.md)", fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


def _video_path(raw: str, base: Path) -> Path:
    """A sessions.csv ``path``: as given if it exists, else relative to the CSV's folder."""
    p = Path(raw)
    return p if p.is_absolute() or p.exists() else base / p


def _backend_kwargs(name: str, cfg: PipelineConfig) -> dict:
    if name == "mediapipe":
        return {"model_path": cfg.pose.mediapipe_model}
    if name == "rtmpose":
        return {"mode": cfg.pose.rtmpose_mode}
    raise typer.BadParameter(f"--backend must be mediapipe or rtmpose, got {name!r}")


@app.command()
def extract(
    session: str = typer.Option(..., help="session id: every clip of it in --sessions"),
    backend: str | None = typer.Option(None, help="mediapipe | rtmpose (default: config)"),
    sessions: str = typer.Option(
        "data/sessions.csv", help="sessions CSV; npz go to <its dir>/interim/<session>/"
    ),
    clip: str | None = typer.Option(None, help="only this clip id"),
    allow_low_fps: bool = typer.Option(False, help="accept clips below capture.min_fps"),
    overwrite: bool = typer.Option(False, help="re-extract clips that already have an npz"),
    config: str = CFG,
) -> None:
    """video → KeypointSeq npz for every clip in a session (TASKS M0).

    Writes ``<sessions dir>/interim/<session>/<clip>.<backend>.npz``, which ``process --session``
    reads. Frame times are container PTS (io.video); the npz meta records the video's sha256,
    its VFR flag and codec. Standing clips are extracted too (foot-angle offset).
    """
    from gaitlab.data.sessions import load_sessions
    from gaitlab.io.video import LowFpsError, check_fps, iter_frames, probe
    from gaitlab.pose import base as pose_base
    from gaitlab.pose.mediapipe_backend import file_sha256

    cfg = load_pipeline_config(config)
    name = backend or cfg.pose.backend
    kwargs = _backend_kwargs(name, cfg)
    rows = [r for r in load_sessions(sessions) if r.session_id == session]
    if not rows:
        raise typer.BadParameter(f"no rows for session {session} in {sessions}")
    if clip is not None:
        rows = [r for r in rows if r.clip_id == clip]
        if not rows:
            raise typer.BadParameter(f"session {session} has no clip {clip!r}")
    base = Path(sessions).parent
    out_dir = base / "interim" / session
    # check every clip before the slow part, so a bad row fails in seconds, not mid-session
    todo = []
    for r in rows:
        video = _video_path(r.path, base)
        if video.suffix == ".npz":
            raise typer.BadParameter(
                f"{r.clip_id}: {r.path} is already keypoints; use `gaitlab process`"
            )
        if not video.exists():
            raise typer.BadParameter(f"{r.clip_id}: video {video} not found")
        info = probe(str(video))
        try:
            check_fps(info, cfg.capture, allow_low_fps)
        except LowFpsError as e:
            raise typer.BadParameter(str(e)) from e
        if abs(info.fps - r.fps) > 0.01 * r.fps:
            typer.secho(f"{r.clip_id}: sessions.csv fps {r.fps} != video {info.fps:.2f}",
                        fg=typer.colors.YELLOW)  # fmt: skip
        if info.vfr:
            typer.secho(f"{r.clip_id}: variable frame rate; timing uses PTS (ADR-0010)",
                        fg=typer.colors.YELLOW)  # fmt: skip
        todo.append((r, video, info))
    est = None
    for r, video, info in todo:
        dest = out_dir / f"{r.clip_id}.{name}.npz"
        if dest.exists() and not overwrite:
            typer.echo(f"skip {dest} (exists; --overwrite to redo)")
            continue
        est = est or pose_base.get_backend(name, **kwargs)
        seq = est.run(iter_frames(str(video)), info.fps, info.size)
        seq.meta.update(
            session_id=session, clip_id=r.clip_id, video=video.name,
            video_sha256=file_sha256(video), vfr=bool(info.vfr), codec=info.codec,
        )  # fmt: skip
        dest.parent.mkdir(parents=True, exist_ok=True)
        seq.save(str(dest))
        n_missing = int((~np.isfinite(seq.xy).all(axis=(1, 2))).sum())
        typer.echo(f"{r.clip_id}: {seq.n_frames} frames ({n_missing} without a person) → {dest}")


def _clip_ids(clip: Path) -> tuple[str, str]:
    """``<dir>/<session>/<clip>.<backend>.npz`` → (session id, clip id)."""
    return clip.parent.name, clip.name.split(".")[0]


def _condition_from_clip_id(clip_id: str) -> str:
    """``natural`` / ``natural-2`` → ``natural``; anything else → ``""``."""
    from typing import get_args

    from gaitlab.data.sessions import Condition

    cond = clip_id.split("-")[0]
    return cond if cond in get_args(Condition) else ""


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
    from gaitlab.data.analysis import analyze_session, clip_rows
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
        strides, windows = clip_rows(seq, cfg, sid, cid, cond, speed_kmh, standing_seq)
        root = Path("data/synthetic") if seq.synthetic else Path("data")
        dest = Path(out) if out else root / "processed" / sid
        _write_processed(dest, f"{cid}.", strides, windows, seq.synthetic)
        return

    from gaitlab.data.sessions import load_sessions

    rows = [r for r in load_sessions(sessions) if r.session_id == session]
    if not rows:
        raise typer.BadParameter(f"no rows for session {session} in {sessions}")
    base = Path(sessions).parent
    done = analyze_session(rows, base / "interim" / session, backend, cfg)
    strides_out, windows_out, flags = done.strides, done.windows, done.synthetic
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
    # The synthetic split is derived from these sessions; a split from an earlier run is stale.
    (o / "splits.json").unlink(missing_ok=True)
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
def label(
    blind: int = typer.Option(0, help="blind re-grade N labeled windows (0 = normal mode)"),
    rater: str = typer.Option("vw", help="rater id written to every row"),
    seed: int = typer.Option(0, help="seed for the blind sample"),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="label gaitlab synth windows → data/synthetic/labels/"
    ),
    root: str | None = typer.Option(None, help="data root (default data, or data/synthetic)"),
    backend: str | None = typer.Option(None, help="npz backend (default: pose.backend)"),
    port: int = typer.Option(7860, help="localhost port"),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="open a browser tab"),
    config: str = CFG,
    features: str = FCFG,
) -> None:
    """Labeling app: grade windows with overlay and features (TECHNICAL_PLAN §6, TASKS M2)."""
    from gaitlab.label import app as label_app

    data_root = Path(root) if root else Path("data/synthetic" if synthetic else "data")
    npz_backend = backend or (
        "synthetic" if synthetic else load_pipeline_config(config).pose.backend
    )
    try:
        ctx = label_app.prepare(
            data_root, synthetic=synthetic, backend=npz_backend, rater=rater, blind=blind,
            seed=seed, pipeline_config=config, features_config=features,
        )  # fmt: skip
    except (ValueError, FileNotFoundError) as e:
        raise typer.BadParameter(str(e)) from e
    label_app.launch(ctx, port=port, open_browser=browser)


SYN_ROOT = typer.Option("data/synthetic", "--root", help="gaitlab synth output folder")
SYNTHETIC_ONLY = typer.Option(
    False, "--synthetic", help="run on gaitlab synth output (required until real labels exist)"
)


def _require_synthetic(synthetic: bool, what: str) -> None:
    if not synthetic:
        typer.secho(
            f"{what} on real labels needs data/labels/labels.csv and data/splits.json (M2). "
            "Use --synthetic for the dry run.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2)


@app.command()
def baselines(
    synthetic: bool = SYNTHETIC_ONLY, root: str = SYN_ROOT, config: str = CFG, features: str = FCFG
) -> None:
    """Rules / logistic regression / LightGBM: fit on train+dev sessions, predict test."""
    import pandas as pd

    from gaitlab.data.synthetic import load_synthetic_grader_inputs, synthetic_features_config
    from gaitlab.eval.run_eval import PRED_COLUMNS
    from gaitlab.models.baselines import baseline_predictions

    _require_synthetic(synthetic, "baselines")
    cfg = load_pipeline_config(config)
    fcfg = synthetic_features_config(load_features_config(features))
    windows, labels, split = load_synthetic_grader_inputs(root)
    preds, meta = baseline_predictions(
        windows, labels, split, fcfg, fcfg.feature_order, synthetic=True, seed=cfg.seed
    )
    out = Path(root) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in preds.items():
        pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(out / f"{name}.csv", index=False)
        if name in meta:
            (out / f"{name}.meta.json").write_text(json.dumps(meta[name], indent=2) + "\n")
    typer.echo(
        f"{', '.join(preds)}: {meta['n_test_windows']} test windows "
        f"({', '.join(meta['test_sessions'])}), fit on {meta['n_fit_windows']} → {out} (synthetic)"
    )


@app.command("build-sft")
def build_sft(
    synthetic: bool = SYNTHETIC_ONLY, root: str = SYN_ROOT, config: str = CFG, features: str = FCFG
) -> None:
    """labels + windows → SFT JSONL by session split (TASKS M3)."""
    from gaitlab.data.synthetic import load_synthetic_grader_inputs
    from gaitlab.llm.sft_data import build_records, write_jsonl

    _require_synthetic(synthetic, "build-sft")
    fcfg = load_features_config(features)
    windows, labels, split = load_synthetic_grader_inputs(root)
    records = build_records(windows, labels, split, fcfg.feature_order, synthetic=True)
    out = Path(root) / "sft"
    for part, recs in records.items():
        write_jsonl(recs, out / f"{part}.jsonl")
    counts = ", ".join(f"{k} {len(v)}" for k, v in records.items())
    typer.echo(f"SFT records: {counts} → {out} (synthetic)")


LORA_CFG = typer.Option("configs/train_lora.yaml", "--lora-config", help="LoRA / grader config")
DEVICE = typer.Option(None, "--device", help="cuda | mps | cpu (default: cuda > mps > cpu)")


@app.command()
def train(
    model: str = typer.Option("1b", help="1b | 270m"),
    sft_dir: str = typer.Option("data/sft", "--sft-dir"),
    allow_synthetic: bool = typer.Option(
        False, "--allow-synthetic", help="dry run on synthetic SFT records"
    ),
    max_steps: int | None = typer.Option(None, help="cap optimizer steps (smoke runs)"),
    out_root: str = typer.Option("runs", "--out-root"),
    device: str | None = DEVICE,
    lora_config: str = LORA_CFG,
) -> None:
    """LoRA fine-tune Gemma 3 on SFT JSONL → runs/<run_id>/ (TECHNICAL_PLAN §7.2)."""
    from gaitlab.llm.train_lora import check_synthetic, read_sft
    from gaitlab.llm.train_lora import train as train_lora

    try:
        check_synthetic(read_sft(sft_dir), allow_synthetic)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(code=2) from e
    run = train_lora(
        model, sft_dir=sft_dir, cfg_path=lora_config, out_root=out_root,
        allow_synthetic=allow_synthetic, max_steps=max_steps, device=device,
    )  # fmt: skip
    meta = json.loads((run / "run.json").read_text())
    flag = " (synthetic)" if meta["synthetic"] else ""
    typer.echo(f"{meta['model_id']} on {meta['device']}, {meta['global_step']} steps → {run}{flag}")


@app.command()
def predict(
    model: str = typer.Option("1b", help="1b | 270m"),
    synthetic: bool = SYNTHETIC_ONLY,
    root: str = SYN_ROOT,
    adapter: str | None = typer.Option(
        None, help="run dir (runs/<run_id>); default: newest run of --model"
    ),
    zero_shot: bool = typer.Option(False, "--zero-shot", help="no adapter"),
    runs_root: str = typer.Option("runs", "--runs", help="where gaitlab train writes runs"),
    consistency_runs: int | None = typer.Option(
        None, help="G6 repeats (default: config consistency.runs; 0 skips)"
    ),
    batch_size: int = typer.Option(8, help="prompts per forward / generate batch"),
    device: str | None = DEVICE,
    lora_config: str = LORA_CFG,
    config: str = CFG,
    features: str = FCFG,
) -> None:
    """Gemma grade (logits) + greedy tags/cue for the test windows → predictions/<model>.csv."""
    import pandas as pd

    from gaitlab.config import config_hash, load_train_lora_config
    from gaitlab.data.synthetic import load_synthetic_grader_inputs
    from gaitlab.eval.run_eval import PRED_COLUMNS
    from gaitlab.llm.grade import llm_model_name, load_grader, predict_windows
    from gaitlab.llm.train_lora import latest_run

    _require_synthetic(synthetic, "predict")
    lcfg = load_train_lora_config(lora_config)
    cfg = load_pipeline_config(config)
    fcfg = load_features_config(features)
    if model not in lcfg.models:
        raise typer.BadParameter(f"--model must be one of {sorted(lcfg.models)}")
    run_meta: dict = {}
    adapter_path = None
    if not zero_shot:
        run_dir = Path(adapter) if adapter else latest_run(runs_root, model, synthetic=True)
        run_meta = json.loads((run_dir / "run.json").read_text())
        if run_meta["model_key"] != model:
            raise typer.BadParameter(f"{run_dir} was trained for {run_meta['model_key']}")
        adapter_path = str(run_dir / "adapter")
    windows, labels, split = load_synthetic_grader_inputs(root)
    test = sorted(
        (w for w in windows if str(w["window_id"]) in labels
         and split.get(str(w["session_id"])) == "test"),
        key=lambda w: str(w["window_id"]),
    )  # fmt: skip
    grader = load_grader(lcfg.models[model], adapter_path, cfg=lcfg, device=device)
    n_rep = lcfg.consistency.runs if consistency_runs is None else consistency_runs
    rows, reps = predict_windows(
        grader, test, fcfg.feature_order, synthetic=True, consistency_runs=n_rep,
        temperature=lcfg.consistency.temperature, seed=cfg.seed, batch_size=batch_size,
    )  # fmt: skip
    name = llm_model_name(model, zero_shot)
    out = Path(root) / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(out / f"{name}.csv", index=False)
    meta = {
        "kind": "llm", "generates_cues": True, "model_key": model, "zero_shot": zero_shot,
        "run_id": run_meta.get("run_id"), "train_config_hash": run_meta.get("config_hash"),
        "lora_config_hash": config_hash(lcfg), **grader.meta(), "synthetic": True,
    }  # fmt: skip
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    cons = out / f"{name}.consistency.json"
    if reps is not None:
        cons.write_text(json.dumps(reps) + "\n")
    else:
        cons.unlink(missing_ok=True)
    typer.echo(
        f"{name}: {len(rows)} test windows, grade tokens {grader.tokens.form!r}, "
        f"device {grader.device} → {out / f'{name}.csv'} (synthetic)"
    )


@app.command("eval")
def eval_(
    synthetic: bool = SYNTHETIC_ONLY,
    root: str = SYN_ROOT,
    results: str = typer.Option("results", help="results root"),
    n_boot: int | None = typer.Option(None, help="bootstrap resamples (default: config)"),
    config: str = CFG,
    features: str = FCFG,
) -> None:
    """Grader metrics for every model's test predictions → results/[synthetic/]grader."""
    from gaitlab.config import config_hash
    from gaitlab.data.synthetic import load_synthetic_grader_inputs, synthetic_features_config
    from gaitlab.eval.run_eval import run

    _require_synthetic(synthetic, "eval")
    cfg = load_pipeline_config(config)
    fcfg = load_features_config(features)
    # Also writes <root>/splits.json if it isn't there yet.
    windows, _, _ = load_synthetic_grader_inputs(root)
    vcfg = synthetic_features_config(fcfg)  # the verifier's thresholds on synthetic data
    r = Path(root)
    paths = run(
        str(r / "predictions"), str(r / "labels.csv"), str(r / "splits.json"), synthetic=True,
        tag_vocab=list(fcfg.tags), n_boot=n_boot or cfg.bootstrap.n, seed=cfg.seed,
        config={"pipeline": config_hash(cfg), "features": config_hash(fcfg)},
        results_root=results, windows={str(w["window_id"]): w for w in windows},
        features_cfg=vcfg,
    )  # fmt: skip
    for p in paths.values():
        typer.echo(f"→ {p}")


@app.command()
def validate(
    target: str = typer.Argument(..., help="own | biocv | mocap"),
    sessions: str = typer.Option("data/sessions.csv", help="own: sessions CSV; npz from interim/"),
    backend: str | None = typer.Option(None, help="own: pose backend (default: config)"),
    other: str | None = typer.Option(None, help="own: second backend for V7 agreement"),
    session: list[str] | None = ONLY_SESSIONS,
    synthetic: bool = typer.Option(False, help="own: on gaitlab synth keypoints"),
    n_boot: int | None = typer.Option(None, help="bootstrap resamples (default: config)"),
    results: str = typer.Option("results", help="results root"),
    config: str = CFG,
) -> None:
    """Validation. ``own``: V5 directions, V6 stability and (with --other) V7 backend agreement
    on your sessions → results/validation. ``biocv`` / ``mocap``: public data (TASKS M0)."""
    if target != "own":
        _todo(f"validate {target}")
    from gaitlab.eval.own_validation import run

    backend = backend or load_pipeline_config(config).pose.backend
    try:
        out = run(sessions, backend=backend, other=other, synthetic=synthetic,
                  config_path=config, n_boot=n_boot, session_ids=session or None,
                  results_root=results)  # fmt: skip
    except (ValueError, FileNotFoundError) as e:
        raise typer.BadParameter(str(e)) from e
    for path in out.values():
        typer.echo(f"→ {path}")


@app.command()
def bench(
    trt: bool = typer.Option(False, "--trt", help="B2: TensorRT vs eager grade scoring pass"),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="dry run on synthetic SFT prompts → results/synthetic/bench"
    ),
    model: str = typer.Option("1b", help="1b | 270m"),
    sft_dir: str | None = typer.Option(
        None, "--sft-dir", help="default data/sft (data/synthetic/sft with --synthetic)"
    ),
    adapter: str | None = typer.Option(None, help="run dir; default: newest matching run"),
    zero_shot: bool = typer.Option(False, "--zero-shot", help="no adapter"),
    runs_root: str = typer.Option("runs", "--runs"),
    bench_config: str = typer.Option("configs/bench.yaml", "--bench-config"),
    lora_config: str = LORA_CFG,
    results: str = typer.Option("results", help="results root"),
) -> None:
    """Benchmarks (TASKS M4). ``--trt``: B2 on the SFT test prompts → results/[synthetic/]bench."""
    from gaitlab.bench.trt import read_test_prompts

    if not trt:
        _todo("bench latency pipeline (bench.latency.bench_pipeline, B1); use --trt for B2")
    sft = Path(sft_dir or ("data/synthetic/sft" if synthetic else "data/sft"))
    prompts, test_synthetic = read_test_prompts(sft / "test.jsonl")
    if test_synthetic != synthetic:
        msg = (
            f"{sft}/test.jsonl is synthetic; pass --synthetic (results go to results/synthetic)"
            if test_synthetic
            else f"{sft}/test.jsonl is not synthetic; drop --synthetic"
        )
        typer.secho(msg, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    from gaitlab.bench.trt import bench_scoring
    from gaitlab.config import config_hash, load_bench_config, load_train_lora_config
    from gaitlab.llm.grade import llm_model_name, load_grader
    from gaitlab.llm.train_lora import latest_run
    from gaitlab.report.results import manifest_hash, write_result

    lcfg = load_train_lora_config(lora_config)
    bcfg = load_bench_config(bench_config)
    if model not in lcfg.models:
        raise typer.BadParameter(f"--model must be one of {sorted(lcfg.models)}")
    run_meta: dict = {}
    adapter_path = None
    if not zero_shot:
        run_dir = Path(adapter) if adapter else latest_run(runs_root, model, synthetic=synthetic)
        run_meta = json.loads((run_dir / "run.json").read_text())
        if run_meta["model_key"] != model or run_meta["synthetic"] is not synthetic:
            raise typer.BadParameter(f"{run_dir} doesn't match --model/--synthetic")
        adapter_path = str(run_dir / "adapter")
    grader = load_grader(lcfg.models[model], adapter_path, cfg=lcfg, device="cuda")
    metrics = bench_scoring(grader, prompts, bcfg)
    metrics.update(
        model=llm_model_name(model, zero_shot), run_id=run_meta.get("run_id"),
        grader={**grader.meta()},
    )  # fmt: skip
    path = write_result(
        "bench", "trt", metrics,
        config={"bench": config_hash(bcfg), "lora": config_hash(lcfg),
                "train": run_meta.get("config_hash")},
        data_manifest=manifest_hash([sft / "test.jsonl"]), synthetic=synthetic,
        notes="Grade scoring pass only (ADR-0004); cue generation stays eager.", root=results,
    )  # fmt: skip
    typer.echo(f"B2 → {path}" + (" (synthetic)" if synthetic else ""))


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
