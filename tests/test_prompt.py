import math

import pytest

from gaitlab.llm.prompt import GRADE_PREFIX, format_prompt, format_target, parse_output

ORDER = ["cadence_spm", "overstride_L"]


def test_prompt_is_deterministic_and_handles_nan():
    row = {"speed_kmh": 12.0, "cadence_spm_med": 164.04, "cadence_spm_iqr": 2.1,
           "overstride_L_med": math.nan, "overstride_L_iqr": 0.03}
    a = format_prompt(row, ORDER)
    assert a == format_prompt(dict(row), ORDER)
    assert "cadence_spm: 164.0 (iqr 2.1)" in a
    assert "overstride_L: NA (iqr 0.030)" in a
    assert a.splitlines()[1] == "speed_kmh: 12.0"


def test_target_starts_with_grade_prefix_and_roundtrips():
    t = format_target(3, ["overstride", "low_cadence"], "Land closer under your hips.")
    assert t.startswith(f"{GRADE_PREFIX} 3")
    p = parse_output(t)
    assert p.grade == 3 and p.tags == ["overstride", "low_cadence"]
    assert p.cue == "Land closer under your hips."


def test_no_tags_and_bad_grade():
    assert parse_output(format_target(5, [], "")).tags == []
    with pytest.raises(ValueError):
        format_target(6, [], "x")
