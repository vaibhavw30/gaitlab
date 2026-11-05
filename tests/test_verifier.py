import pytest

from gaitlab.llm.verifier import ThresholdsNotCalibratedError, flagged_tags, verify
from gaitlab.models.rules import rules_predict

OVERSTRIDING = {
    "overstride_L_med": 0.40,
    "cadence_spm_med": 168.0,
    "vert_osc_L_med": 0.08,
    "knee_flex_ic_deg_med": 40.0,
    "foot_strike_deg_med": 10.0,
    "trunk_lean_deg_med": 6.0,
}


def test_uncalibrated_thresholds_refuse(features_cfg):
    with pytest.raises(ThresholdsNotCalibratedError):
        flagged_tags(OVERSTRIDING, features_cfg)
    with pytest.raises(ThresholdsNotCalibratedError):
        rules_predict(OVERSTRIDING, features_cfg)


def test_grounded_cue_passes(synthetic_features_cfg):
    r = verify(OVERSTRIDING, ["overstride"], "Land closer under your hips.", synthetic_features_cfg)
    assert r.passed, r.reasons


def test_unsupported_tag_fails(synthetic_features_cfg):
    r = verify(OVERSTRIDING, ["high_bounce"], "Less bounce, please.", synthetic_features_cfg)
    assert not r.passed and "not supported" in r.reasons[0]


def test_cue_must_mention_tag(synthetic_features_cfg):
    r = verify(OVERSTRIDING, ["overstride"], "Great job, keep it up.", synthetic_features_cfg)
    assert not r.passed


def test_unknown_tag_fails(synthetic_features_cfg):
    assert not verify(OVERSTRIDING, ["pelvic_drop"], "x", synthetic_features_cfg).passed


def test_no_tags_with_issue_cue_fails(synthetic_features_cfg):
    clean = dict(OVERSTRIDING, overstride_L_med=0.2)
    assert verify(clean, [], "", synthetic_features_cfg).passed
    assert not verify(clean, [], "Raise your cadence.", synthetic_features_cfg).passed


def test_rules_grade(synthetic_features_cfg):
    grade, tags = rules_predict(OVERSTRIDING, synthetic_features_cfg)
    assert tags == ["overstride"] and grade == 4


def test_failure_reasons_have_stable_codes(synthetic_features_cfg):
    cfg = synthetic_features_cfg
    clean = dict(OVERSTRIDING, overstride_L_med=0.2)
    assert verify(OVERSTRIDING, ["overstride"], "Land closer.", cfg).codes == []
    assert verify(OVERSTRIDING, ["pelvic_drop"], "x", cfg).codes == ["unknown_tag"]
    assert verify(OVERSTRIDING, ["high_bounce"], "Less bounce.", cfg).codes == ["unsupported_tag"]
    assert verify(OVERSTRIDING, ["overstride"], "Great job.", cfg).codes == ["cue_misses_tags"]
    assert verify(clean, [], "Raise your cadence.", cfg).codes == ["cue_implies_untagged"]
    r = verify(OVERSTRIDING, ["high_bounce", "pelvic_drop"], "Nice.", cfg)
    assert r.codes == ["unknown_tag", "unsupported_tag", "cue_misses_tags"]
    assert len(r.reasons) == len(r.codes)
