"""Labeling (TECHNICAL_PLAN §6, US-5, FR-7; TASKS M2): queues, blind re-grade, append-only CSV.

The logic is pure (``label.queue``) and the CSV writer lives in ``data.sessions``, so all of it
runs without Gradio. The Gradio wiring is checked by building the app, not launching it.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import multiprocessing as mp
from pathlib import Path

import pytest

from gaitlab.data.sessions import LABEL_FIELDS, LabelRow, append_label, load_labels
from gaitlab.label.queue import (
    LabelSession,
    blind_sample,
    check_synthetic,
    feature_rows,
    make_label,
    normal_queue,
)

VOCAB = ["overstride", "low_cadence", "high_bounce"]
NOW = dt.datetime(2026, 11, 11, 12, 0, tzinfo=dt.UTC)


def _w(sid: str, cid: str, start: int, t: float, **feat) -> dict:
    return {"window_id": f"{sid}:{cid}:{start}", "session_id": sid, "clip_id": cid,
            "t_start": t, "t_end": t + 10.0, "synthetic": False, **feat}  # fmt: skip


def _label(wid: str, grade: int = 3, mode: str = "normal", minute: int = 0) -> LabelRow:
    return LabelRow(window_id=wid, grade=grade, rater="vw", mode=mode,
                    labeled_at=NOW + dt.timedelta(minutes=minute))  # fmt: skip


WINDOWS = [
    _w("20261106-01", "natural", 20, 14.0),
    _w("20261105-01", "overstride", 0, 0.5),
    _w("20261105-01", "natural", 20, 14.0),
    _w("20261105-01", "natural", 0, 0.2),
]


# --- append-only CSV --------------------------------------------------------------------------


def test_append_writes_the_header_once_and_rows_that_load_back(tmp_path: Path):
    path = tmp_path / "labels" / "labels.csv"
    a = make_label("s:natural:0", 4, ["overstride"], " land\n closer ", "vw", "normal", VOCAB, NOW)
    append_label(path, a)
    append_label(path, make_label("s:natural:20", 2, [], "", "vw", "normal", VOCAB, NOW))
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(LABEL_FIELDS) and len(lines) == 3
    back = load_labels(path)
    assert back[0] == a and back[0].cue == "land closer" and back[0].tags == "overstride"
    assert back[1].grade == 2


def test_earlier_rows_are_never_rewritten(tmp_path: Path):
    path = tmp_path / "labels.csv"
    append_label(path, _label("s:a:0", 3))
    before = path.read_bytes()
    append_label(path, _label("s:a:0", 5, minute=1))  # a correction is a new row; later wins
    assert path.read_bytes().startswith(before)


def test_a_file_with_another_header_is_refused(tmp_path: Path):
    path = tmp_path / "labels.csv"
    path.write_text("window_id,grade\nx,3\n")
    with pytest.raises(ValueError, match="header"):
        append_label(path, _label("s:a:0"))
    assert path.read_text() == "window_id,grade\nx,3\n"


def _append_many(args: tuple[str, int]) -> None:
    path, k = args
    for i in range(25):
        append_label(Path(path), _label(f"s:p{k}:{i}", 1 + i % 5))


def test_concurrent_appends_keep_every_row_whole(tmp_path: Path):
    """The file lock: four writers, 100 rows, one header, every row parses."""
    path = tmp_path / "labels.csv"
    with mp.get_context("spawn").Pool(4) as pool:
        pool.map(_append_many, [(str(path), k) for k in range(4)])
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100 and path.read_text().count("window_id,grade") == 1
    assert len(load_labels(path)) == 100


# --- label rows -------------------------------------------------------------------------------


def test_make_label_validates_grade_tags_and_mode():
    row = make_label("s:a:0", 5, ["low_cadence", "overstride"], "", "vw", "normal", VOCAB, NOW)
    assert row.tags == "overstride;low_cadence"  # vocabulary order, not click order
    assert row.labeled_at == NOW
    with pytest.raises(ValueError, match="grade"):
        make_label("s:a:0", None, [], "", "vw", "normal", VOCAB, NOW)
    with pytest.raises(ValueError, match="grade"):
        make_label("s:a:0", 6, [], "", "vw", "normal", VOCAB, NOW)
    with pytest.raises(ValueError, match="unknown tag"):
        make_label("s:a:0", 3, ["slouch"], "", "vw", "normal", VOCAB, NOW)
    with pytest.raises(ValueError, match="rater"):
        make_label("s:a:0", 3, [], "", " ", "normal", VOCAB, NOW)
    blind = make_label("s:a:0", "2", [], "", "vw", "blind_regrade", VOCAB, NOW)
    assert blind.mode == "blind_regrade" and blind.grade == 2


# --- queues -----------------------------------------------------------------------------------


def test_normal_queue_is_unlabeled_windows_by_session_then_clip_then_time():
    labels = [
        _label("20261105-01:natural:20"),
        _label("20261105-01:natural:0", mode="blind_regrade"),
    ]
    assert normal_queue(WINDOWS, labels) == [
        "20261105-01:overstride:0",
        "20261105-01:natural:0",  # only a blind row: still needs its normal label
        "20261106-01:natural:20",
    ]
    assert normal_queue(WINDOWS, []) [:3] == [
        "20261105-01:overstride:0", "20261105-01:natural:0", "20261105-01:natural:20",
    ]  # fmt: skip


def test_blind_sample_is_seeded_from_normal_labels_and_resumes():
    labels = [_label(f"s:c:{i}") for i in range(10)]
    a = blind_sample(labels, 4, seed=0)
    assert len(a) == 4 == len(set(a)) and a == blind_sample(labels, 4, seed=0)
    assert a != blind_sample(labels, 4, seed=1)
    assert len(blind_sample(labels, 40, seed=0)) == 10  # capped at what is labeled
    done = [*labels, _label(a[0], mode="blind_regrade", minute=5)]
    assert blind_sample(done, 4, seed=0) == a[1:]  # a restart skips what was re-graded
    assert blind_sample([_label("s:c:0", mode="blind_regrade")], 4, seed=0) == []
    with pytest.raises(ValueError):
        blind_sample(labels, 0, seed=0)


def test_synthetic_windows_only_go_to_the_synthetic_labels():
    check_synthetic({False}, synthetic_mode=False)
    check_synthetic({True}, synthetic_mode=True)
    with pytest.raises(ValueError, match="--synthetic"):
        check_synthetic({True}, synthetic_mode=False)
    with pytest.raises(ValueError, match="not synthetic"):
        check_synthetic({False}, synthetic_mode=True)
    with pytest.raises(ValueError, match="mix"):
        check_synthetic({True, False}, synthetic_mode=False)
    with pytest.raises(ValueError, match="no windows"):
        check_synthetic(set(), synthetic_mode=False)


def test_feature_rows_show_median_and_iqr_in_feature_order():
    w = _w("s", "c", 0, 0.0, cadence_spm_med=170.0, cadence_spm_iqr=2.0, vert_osc_L_med=0.08)
    rows = feature_rows(w, ["cadence_spm", "vert_osc_L", "trunk_lean_deg"])
    assert rows == [["cadence_spm", 170.0, 2.0], ["vert_osc_L", 0.08, None],
                    ["trunk_lean_deg", None, None]]  # fmt: skip


# --- the session state behind the UI ----------------------------------------------------------


def _session(mode: str = "normal", ids=None):
    written: list[LabelRow] = []
    by_id = {w["window_id"]: w for w in WINDOWS}
    s = LabelSession(ids or list(by_id), by_id, mode=mode, rater="vw", vocab=VOCAB,
                     write=written.append, now=lambda: NOW)  # fmt: skip
    return s, written


def test_save_writes_and_advances_skip_and_back_only_move():
    s, written = _session(ids=["a", "b", "c"])
    s.windows.update({k: _w("s", k, 0, 0.0) | {"window_id": k} for k in "abc"})
    assert s.current() == "a" and s.progress().startswith("1 / 3")
    s.save(4, ["overstride"], "shorter stride")
    assert [r.window_id for r in written] == ["a"] and s.current() == "b"
    s.skip()
    assert s.current() == "c" and len(written) == 1
    s.back()
    assert s.current() == "b"
    s.save(2, [], "")
    s.save(3, [], "")
    assert s.done and s.current() is None and [r.window_id for r in written] == ["a", "b", "c"]
    assert "3 saved" in s.progress()
    with pytest.raises(ValueError, match="no window"):
        s.save(3, [], "")


def test_a_bad_save_writes_nothing_and_stays_put():
    s, written = _session()
    first = s.current()
    with pytest.raises(ValueError):
        s.save(None, [], "")
    assert written == [] and s.current() == first


def test_blind_session_writes_blind_rows_and_shows_no_window_id_or_features():
    s, written = _session(mode="blind_regrade", ids=["20261105-01:overstride:0"])
    assert s.blind and s.title() == "Blind re-grade 1 / 1"
    assert "overstride" not in s.title() and s.features(["cadence_spm"]) == []
    s.save(1, [], "")
    assert written[0].mode == "blind_regrade"
    n, _ = _session(ids=["20261105-01:overstride:0"])
    assert "20261105-01:overstride:0" in n.title()
    with pytest.raises(ValueError, match="mode"):
        _session(mode="regrade")


# --- the Gradio app (built, not launched) -----------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("gradio") is None, reason="needs the 'app' extra")
def test_the_app_builds_from_processed_synthetic_windows(tmp_path: Path):
    from typer.testing import CliRunner

    from gaitlab.cli import app as cli
    from gaitlab.label.app import build_app, prepare

    root = tmp_path / "synthetic"
    runner = CliRunner()
    assert runner.invoke(cli, ["synth", "--n-sessions", "1", "--out", str(root),
                               "--write-keypoints"]).exit_code == 0  # fmt: skip
    sid = next(p.name for p in (root / "interim").iterdir())
    res = runner.invoke(cli, ["process", "--session", sid, "--sessions",
                              str(root / "sessions.csv"), "--backend", "synthetic"])  # fmt: skip
    assert res.exit_code == 0, res.output
    with pytest.raises(ValueError, match="--synthetic"):
        prepare(root, synthetic=False, backend="synthetic")
    ctx = prepare(root, synthetic=True, backend="synthetic", rater="vw")
    assert ctx.labels_csv == root / "labels" / "hand_labels.csv"  # never synth's rule labels.csv
    assert ctx.session.current() is not None
    first = ctx.session.current()
    path = ctx.overlay(first)
    assert path.exists() and path.suffix == ".mp4" and path.is_relative_to(root / "labels")
    assert ctx.overlay(first) == path  # cached
    demo = build_app(ctx)
    assert type(demo).__name__ == "Blocks"


def test_label_cli_prepares_and_launches_with_the_right_root(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    from gaitlab.cli import app as cli

    seen: dict = {}

    def fake_prepare(root, **kw):
        seen.update(root=Path(root), **kw)
        return "ctx"

    monkeypatch.setattr("gaitlab.label.app.prepare", fake_prepare)
    monkeypatch.setattr("gaitlab.label.app.launch", lambda ctx, **kw: seen.update(launched=kw))
    runner = CliRunner()
    res = runner.invoke(cli, ["label", "--blind", "40", "--rater", "vw", "--no-browser"])
    assert res.exit_code == 0, res.output
    assert seen["root"] == Path("data") and seen["blind"] == 40 and seen["synthetic"] is False
    assert seen["backend"] == "mediapipe" and seen["launched"]["open_browser"] is False
    res = runner.invoke(cli, ["label", "--synthetic"])
    assert res.exit_code == 0, res.output
    assert seen["root"] == Path("data/synthetic") and seen["backend"] == "synthetic"

    def refuse(root, **kw):
        raise ValueError("these windows are synthetic: use --synthetic")

    monkeypatch.setattr("gaitlab.label.app.prepare", refuse)
    res = runner.invoke(cli, ["label"])
    assert res.exit_code != 0 and "--synthetic" in res.output
