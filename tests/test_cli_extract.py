"""gaitlab extract: video → KeypointSeq npz per clip of a session (TASKS M0 "CLI wiring").

Clips are generated in-test with PyAV and the pose backend is a fake, so no model, mediapipe or
rtmlib is needed. The real backends have their own tests (test_*_backend.py).
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gaitlab.cli import app
from gaitlab.data.sessions import SessionRow
from gaitlab.pose.mediapipe_backend import file_sha256
from gaitlab.pose.skeleton import CANONICAL
from gaitlab.types import KeypointSeq

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("av") is None, reason="needs the 'video' extra (PyAV)"
)
ROOT = Path(__file__).resolve().parents[1]
CFG = str(ROOT / "configs" / "default.yaml")
SID = "20261105-01"
runner = CliRunner()


def _clip(path: Path, n: int = 6, fps: int = 120, w: int = 64, h: int = 48) -> Path:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            img = np.full((h, w, 3), 16 + 20 * i, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def _sessions(path: Path, rows: list[dict]) -> Path:
    fields = list(SessionRow.model_fields)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fields})
    return path


def _row(clip_id: str, path: Path, condition: str = "natural", fps: float = 120.0) -> dict:
    return {"session_id": SID, "date": "2026-11-05", "clip_id": clip_id, "path": str(path),
            "condition": condition, "speed_kmh": 10.0, "fps": fps}  # fmt: skip


class FakeBackend:
    name = "mediapipe"

    def __init__(self) -> None:
        self.calls: list[tuple[int, float, tuple[int, int]]] = []

    def run(self, frames, fps, size):
        frames = list(frames)
        self.calls.append((len(frames), fps, size))
        n = len(frames)
        return KeypointSeq(
            xy=np.zeros((n, len(CANONICAL), 2), np.float32),
            vis=np.ones((n, len(CANONICAL)), np.float32),
            t=np.array([t for t, _ in frames]), fps=fps, size=size, backend=self.name,
            meta={"mediapipe": "fake"},
        )  # fmt: skip


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> dict:
    made: dict = {}

    def get_backend(name: str, **kwargs):
        made["name"], made["kwargs"] = name, kwargs
        made["backend"] = b = FakeBackend()
        b.name = name
        return b

    monkeypatch.setattr("gaitlab.pose.base.get_backend", get_backend)
    return made


def _extract(sessions: Path, *extra: str):
    return runner.invoke(
        app, ["extract", "--session", SID, "--sessions", str(sessions), "--config", CFG, *extra]
    )


def test_extract_writes_one_npz_per_clip_where_process_reads_them(tmp_path: Path, fake: dict):
    raw = tmp_path / "raw" / SID
    nat, stand = _clip(raw / "natural.mp4"), _clip(raw / "standing.mp4", n=4)
    sessions = _sessions(tmp_path / "sessions.csv",
                         [_row("natural", nat), _row("standing", stand, "standing")])  # fmt: skip
    res = _extract(sessions)
    assert res.exit_code == 0, res.output
    out = tmp_path / "interim" / SID
    seq = KeypointSeq.load(str(out / "natural.mediapipe.npz"))
    assert seq.n_frames == 6 and not seq.synthetic and seq.size == (64, 48)
    assert np.all(np.diff(seq.t) > 0)  # PTS times from io.video
    assert seq.meta["video_sha256"] == file_sha256(nat) and seq.meta["vfr"] is False
    assert seq.meta["session_id"] == SID and seq.meta["clip_id"] == "natural"
    assert (out / "standing.mediapipe.npz").exists()  # standing clips feed the foot-angle offset
    assert fake["name"] == "mediapipe"
    assert fake["kwargs"] == {"model_path": "models/pose_landmarker_heavy.task"}


def test_rtmpose_gets_its_mode_from_the_config(tmp_path: Path, fake: dict):
    nat = _clip(tmp_path / "raw" / SID / "natural.mp4")
    res = _extract(_sessions(tmp_path / "sessions.csv", [_row("natural", nat)]),
                   "--backend", "rtmpose")  # fmt: skip
    assert res.exit_code == 0, res.output
    assert fake["kwargs"] == {"mode": "performance"}
    assert (tmp_path / "interim" / SID / "natural.rtmpose.npz").exists()


def test_low_fps_clip_is_refused_unless_allowed(tmp_path: Path, fake: dict):
    nat = _clip(tmp_path / "raw" / SID / "natural.mp4", fps=30)
    sessions = _sessions(tmp_path / "sessions.csv", [_row("natural", nat, fps=30.0)])
    res = _extract(sessions)
    assert res.exit_code != 0 and "fps" in res.output
    assert not (tmp_path / "interim").exists()
    assert _extract(sessions, "--allow-low-fps").exit_code == 0


def test_existing_npz_is_kept_unless_overwrite(tmp_path: Path, fake: dict):
    nat = _clip(tmp_path / "raw" / SID / "natural.mp4")
    sessions = _sessions(tmp_path / "sessions.csv", [_row("natural", nat)])
    assert _extract(sessions).exit_code == 0
    first = fake["backend"]
    res = _extract(sessions)
    assert res.exit_code == 0 and "skip" in res.output
    assert fake["backend"] is first, "no model is loaded when every clip is skipped"
    assert _extract(sessions, "--overwrite").exit_code == 0
    assert fake["backend"] is not first and len(fake["backend"].calls) == 1


def test_only_one_clip_with_clip_option(tmp_path: Path, fake: dict):
    raw = tmp_path / "raw" / SID
    rows = [_row("natural", _clip(raw / "natural.mp4")), _row("cue", _clip(raw / "cue.mp4"))]
    res = _extract(_sessions(tmp_path / "sessions.csv", rows), "--clip", "cue")
    assert res.exit_code == 0, res.output
    assert sorted(p.name for p in (tmp_path / "interim" / SID).iterdir()) == ["cue.mediapipe.npz"]


def test_keypoint_rows_unknown_sessions_and_missing_videos_are_refused(tmp_path: Path, fake):
    npz_row = _row("natural", tmp_path / "interim" / SID / "natural.synthetic.npz")
    res = _extract(_sessions(tmp_path / "a.csv", [npz_row]))
    assert res.exit_code != 0 and "process" in res.output
    res = _extract(_sessions(tmp_path / "b.csv", [_row("natural", tmp_path / "nope.mp4")]))
    assert res.exit_code != 0 and "nope.mp4" in res.output
    other = _sessions(tmp_path / "c.csv", [{**_row("natural", tmp_path / "x.mp4"),
                                            "session_id": "20261105-02"}])  # fmt: skip
    res = _extract(other)
    assert res.exit_code != 0 and SID in res.output
