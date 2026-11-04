"""CLI wiring for the no-video path: synth --write-keypoints → process → overlay (TASKS M0).

Everything here runs on synthetic keypoints, so every artifact must carry the synthetic flag
(CLAUDE.md rule 3).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from gaitlab.cli import app
from gaitlab.data.sessions import load_sessions
from gaitlab.data.synthetic import CONDITIONS, synth_clip
from gaitlab.types import KeypointSeq

ROOT = Path(__file__).resolve().parents[1]
CFG = str(ROOT / "configs" / "default.yaml")
FCFG = str(ROOT / "configs" / "features.yaml")

requires_pil = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="needs the 'video' extra (Pillow)"
)

runner = CliRunner()


def _invoke(*args: str):
    res = runner.invoke(app, list(args), catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return res


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    _invoke(
        "synth", "--n-sessions", "1", "--out", str(out), "--write-keypoints",
        "--config", CFG, "--features", FCFG,
    )  # fmt: skip
    return out


def test_synth_write_keypoints_saves_every_clip_as_synthetic_npz(synth_dir: Path):
    npz = sorted((synth_dir / "interim").glob("*/*.synthetic.npz"))
    assert {p.name.split(".")[0] for p in npz} == set(CONDITIONS)
    for p in npz:
        seq = KeypointSeq.load(str(p))
        assert seq.backend == "synthetic"
        assert seq.synthetic
        assert seq.meta["synthetic"] is True
    # Windows from synth also carry the flag, and the folder marker is there.
    assert pd.read_parquet(synth_dir / "windows.parquet")["synthetic"].all()
    assert (synth_dir / "SYNTHETIC").exists()


def test_synth_sessions_csv_matches_schema_and_points_at_the_npz(synth_dir: Path):
    rows = load_sessions(synth_dir / "sessions.csv")
    assert {r.clip_id for r in rows} == set(CONDITIONS)
    assert {r.condition for r in rows} == set(CONDITIONS)
    assert len({r.session_id for r in rows}) == 1
    for r in rows:
        assert Path(r.path).exists()
        assert Path(r.path).parent.name == r.session_id
        assert r.fps > 0
    raw = pd.read_csv(synth_dir / "sessions.csv")
    assert raw["synthetic"].all()


def test_process_session_from_synthetic_sessions_csv(synth_dir: Path):
    sid = load_sessions(synth_dir / "sessions.csv")[0].session_id
    _invoke(
        "process", "--session", sid, "--sessions", str(synth_dir / "sessions.csv"),
        "--backend", "synthetic", "--config", CFG,
    )  # fmt: skip
    out = synth_dir / "processed" / sid
    strides = pd.read_parquet(out / "strides.parquet")
    windows = pd.read_parquet(out / "windows.parquet")
    assert len(strides) > 0 and len(windows) > 0
    assert strides["synthetic"].all() and windows["synthetic"].all()
    assert set(windows["clip_id"]) <= set(CONDITIONS)
    assert (out / "SYNTHETIC").exists()


@pytest.fixture
def clip_npz(tmp_path: Path) -> Path:
    seq, _ = synth_clip(duration_s=20.0)
    path = tmp_path / "20260101-01" / "natural.synthetic.npz"
    path.parent.mkdir()
    seq.save(str(path))
    return path


def test_process_clip_writes_strides_and_windows_without_a_sessions_row(
    clip_npz: Path, tmp_path: Path
):
    out = tmp_path / "out"
    _invoke("process", "--clip", str(clip_npz), "--out", str(out), "--config", CFG)
    strides = pd.read_parquet(out / "natural.strides.parquet")
    windows = pd.read_parquet(out / "natural.windows.parquet")
    assert len(strides) > 0 and len(windows) > 0
    # Session from the parent folder, clip from the file name, condition from the clip id.
    assert set(strides["session_id"]) == {"20260101-01"}
    assert set(windows["clip_id"]) == {"natural"}
    assert set(windows["condition"]) == {"natural"}
    assert windows["window_id"].iloc[0].startswith("20260101-01:natural:")
    assert strides["synthetic"].all() and windows["synthetic"].all()
    assert (out / "SYNTHETIC").exists()


def test_process_clip_defaults_synthetic_output_under_data_synthetic(
    clip_npz: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _invoke("process", "--clip", str(clip_npz), "--config", CFG)
    out = tmp_path / "data" / "synthetic" / "processed" / "20260101-01"
    assert (out / "natural.windows.parquet").exists()
    assert not (tmp_path / "data" / "processed").exists()


def test_process_needs_exactly_one_of_session_or_clip(clip_npz: Path):
    for args in ([], ["--session", "20260101-01", "--clip", str(clip_npz)]):
        res = runner.invoke(app, ["process", *args, "--config", CFG])
        assert res.exit_code != 0
        assert "exactly one of --session or --clip" in res.output


@requires_pil
def test_overlay_cli_gif_without_video(tmp_path: Path):
    seq, _ = synth_clip(duration_s=3.0, size=(400, 300), px_per_m=100.0)
    npz = tmp_path / "natural.synthetic.npz"
    seq.save(str(npz))
    _invoke("overlay", "--clip", str(npz), "--gif", "--config", CFG)
    gif = tmp_path / "natural.synthetic.overlay.gif"
    from PIL import Image

    with Image.open(gif) as im:
        assert im.size[0] == 400  # narrower than the 480-px cap, so not resized
        assert im.n_frames > 1


def test_process_strides_carry_condition_and_knee_cycle(synth_dir: Path):
    sid = load_sessions(synth_dir / "sessions.csv")[0].session_id
    strides = pd.read_parquet(synth_dir / "processed" / sid / "strides.parquet")
    assert set(strides["condition"]) == set(CONDITIONS)
    assert all(len(c) == 101 for c in strides["knee_flex_cycle_deg"])


@pytest.fixture(scope="module")
def grader_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 6-session synthetic dataset run through baselines, build-sft and eval."""
    root = tmp_path_factory.mktemp("grader")
    data, results = root / "synthetic", root / "results"
    _invoke("synth", "--n-sessions", "6", "--out", str(data), "--config", CFG, "--features", FCFG)
    common = ("--synthetic", "--root", str(data), "--config", CFG, "--features", FCFG)
    _invoke("baselines", *common)
    _invoke("build-sft", *common)
    _invoke("eval", *common, "--results", str(results), "--n-boot", "50")
    return root


def test_synthetic_split_is_by_session_and_written_beside_the_data(grader_dir: Path):
    split = json.loads((grader_dir / "synthetic" / "splits.json").read_text())
    assert sorted(split.values()).count("test") == 2 and list(split.values()).count("dev") == 1


def test_build_sft_synthetic_writes_session_disjoint_jsonl(grader_dir: Path):
    split = json.loads((grader_dir / "synthetic" / "splits.json").read_text())
    seen: dict[str, set[str]] = {}
    for part in ("train", "dev", "test"):
        lines = (grader_dir / "synthetic" / "sft" / f"{part}.jsonl").read_text().splitlines()
        assert lines, part
        recs = [json.loads(x) for x in lines]
        assert all(r["synthetic"] is True for r in recs)
        seen[part] = {r["window_id"].split(":")[0] for r in recs}
        assert all(split[s] == part for s in seen[part])
    assert not (seen["train"] & seen["test"]) and not (seen["dev"] & seen["test"])


def test_baselines_and_eval_synthetic_write_only_synthetic_results(grader_dir: Path):
    preds = grader_dir / "synthetic" / "predictions"
    assert {p.stem for p in preds.glob("*.csv")} == {"rules", "logreg", "lgbm"}
    out = grader_dir / "results" / "synthetic" / "grader"
    assert {p.stem for p in out.glob("*.json")} == {"ladder", "tags", "calibration"}
    assert not (grader_dir / "results" / "grader").exists()
    for p in out.glob("*.json"):
        assert json.loads(p.read_text())["synthetic"] is True


def test_baselines_without_synthetic_is_not_available_yet():
    res = runner.invoke(app, ["baselines"])
    assert res.exit_code != 0
