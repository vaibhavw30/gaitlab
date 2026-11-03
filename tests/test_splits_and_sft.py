import pytest

from gaitlab.data.splits import leave_one_session_out, make_session_split, write_split_frozen
from gaitlab.llm.sft_data import build_records

SESSIONS = [(f"2026010{i}-01", f"2026-01-0{i}") for i in range(1, 7)]


def test_split_by_date():
    s = make_session_split(SESSIONS)
    assert s["20260106-01"] == "test" and s["20260105-01"] == "test"
    assert s["20260104-01"] == "dev"
    assert sum(v == "train" for v in s.values()) == 3


def test_split_needs_enough_sessions():
    with pytest.raises(ValueError):
        make_session_split(SESSIONS[:3])


def test_frozen_split(tmp_path):
    p = tmp_path / "splits.json"
    s = make_session_split(SESSIONS)
    write_split_frozen(s, p)
    write_split_frozen(s, p)  # identical is fine
    with pytest.raises(FileExistsError):
        write_split_frozen({**s, "20260101-01": "test"}, p)


def test_loso():
    folds = leave_one_session_out(["a", "a", "b", "c"])
    assert len(folds) == 3
    assert folds[0] == ([2, 3], [0, 1])


def test_build_records_routes_by_session():
    windows = [
        {"window_id": "w1", "session_id": "20260101-01", "cadence_spm_med": 160.0},
        {"window_id": "w2", "session_id": "20260106-01", "cadence_spm_med": 170.0},
        {"window_id": "w3", "session_id": "20260106-01", "cadence_spm_med": 171.0},  # unlabeled
    ]
    labels = {
        "w1": {"grade": 3, "tags": "low_cadence", "cue": "Quicker steps."},
        "w2": {"grade": 5, "tags": "", "cue": ""},
    }
    split = make_session_split(SESSIONS)
    out = build_records(windows, labels, split, ["cadence_spm"], synthetic=True)
    assert [r["window_id"] for r in out["train"]] == ["w1"]
    assert [r["window_id"] for r in out["test"]] == ["w2"]
    assert out["train"][0]["synthetic"] is True
    assert out["train"][0]["messages"][1]["content"].startswith("grade: 3")
