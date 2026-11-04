"""data.sessions: sessions.csv loader (TECHNICAL_PLAN §4.6)."""

from __future__ import annotations

from pathlib import Path

from gaitlab.data.sessions import load_sessions

HEADER = (
    "session_id,date,clip_id,path,condition,speed_kmh,incline_pct,shoes,fps,"
    "camera_height_m,camera_distance_m,notes\n"
)


def test_blank_cells_take_field_defaults(tmp_path: Path):
    csv = tmp_path / "sessions.csv"
    csv.write_text(HEADER + "20261105-01,2026-11-05,natural,raw/n.mp4,natural,,,,120,,,\n")
    (row,) = load_sessions(csv)
    assert row.shoes == "" and row.notes == ""
    assert row.speed_kmh is None and row.camera_height_m is None
    assert row.fps == 120.0
