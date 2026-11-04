import statistics as st

import numpy as np
import pytest

from gaitlab.config import PipelineConfig
from gaitlab.data.synthetic import CONDITIONS, GaitParams, NoiseParams, condition_params, synth_clip
from gaitlab.gait import geometry
from gaitlab.gait.features import analyze_clip, stride_features
from gaitlab.types import Events


def med(strides, field):
    return st.median(getattr(s, field) for s in strides)


def truth_strides(seq, truth, cfg):
    """Features from the noise-free clip at the true events (the reference)."""
    p = truth.params
    xy = geometry.to_y_up(truth.xy_clean, seq.size[1])
    ev = Events(p.near_side, truth.ic[p.near_side], truth.to[p.near_side])
    return stride_features(xy, seq.t, p.near_side, p.direction, truth.leg_length_px, ev, cfg, 0.0)


def test_reference_features_match_construction():
    cfg = PipelineConfig()
    p = GaitParams()
    seq, truth = synth_clip(p)
    ref = truth_strides(seq, truth, cfg)
    assert abs(med(ref, "cadence_spm") - p.cadence_spm) < 1.5  # frame quantization
    assert abs(med(ref, "overstride_L") - truth.overstride_L) < 0.03  # IC frame is ≤1 frame late
    assert abs(med(ref, "foot_strike_deg") - p.foot_strike_deg) < 1.5
    assert abs(med(ref, "trunk_lean_deg") - p.trunk_lean_deg) < 1e-6
    assert abs(med(ref, "vert_osc_L") - truth.vert_osc_L) < 0.003
    assert abs(med(ref, "duty_factor") - p.duty_factor) < 0.02


@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_end_to_end_close_to_reference(condition):
    cfg = PipelineConfig()
    p = condition_params(GaitParams(direction=-1), condition)
    seq, truth = synth_clip(p, noise=NoiseParams(jitter_px=3.0, dropout_frac=0.01, seed=3))
    got = analyze_clip(seq, cfg).strides
    ref = truth_strides(seq, truth, cfg)
    tol = {
        "cadence_spm": 1.5,
        "overstride_L": 0.03,
        "foot_strike_deg": 2.0,
        "vert_osc_L": 0.01,
        "trunk_lean_deg": 1.0,
        "contact_time_s": 2.5 / seq.fps,
        # Contact-instant joint angles move fast at 120 fps; a 1-frame event shift costs degrees.
        "knee_flex_ic_deg": 12.0,
        "shin_angle_deg": 7.0,
    }
    for field, t in tol.items():
        assert abs(med(got, field) - med(ref, field)) < t, field


def test_condition_directions():
    """EVALUATION V5 on synthetic data: features move the way each condition should."""
    cfg = PipelineConfig()
    noise = NoiseParams(jitter_px=3.0, seed=7)
    res = {
        c: analyze_clip(synth_clip(condition_params(GaitParams(), c), noise=noise)[0], cfg).strides
        for c in CONDITIONS
    }
    nat = res["natural"]
    assert med(res["metronome"], "cadence_spm") > med(nat, "cadence_spm")
    assert med(res["metronome"], "overstride_L") < med(nat, "overstride_L")
    assert med(res["metronome"], "vert_osc_L") < med(nat, "vert_osc_L")
    assert med(res["overstride"], "overstride_L") > med(nat, "overstride_L")
    assert med(res["overstride"], "foot_strike_deg") > med(nat, "foot_strike_deg")
    assert med(res["lean"], "trunk_lean_deg") > med(nat, "trunk_lean_deg") + 5
    assert med(res["fatigue"], "vert_osc_L") > med(nat, "vert_osc_L")


def test_standing_calibration_flag():
    cfg = PipelineConfig()
    seq, _ = synth_clip(GaitParams())
    standing, _ = synth_clip(GaitParams(), duration_s=3, standing=True)
    uncal = analyze_clip(seq, cfg).strides
    cal = analyze_clip(seq, cfg, standing=standing).strides
    assert not uncal[0].foot_strike_calibrated and cal[0].foot_strike_calibrated
    # Standing foot is flat in the synthetic model, so the offset is ~0.
    assert abs(med(cal, "foot_strike_deg") - med(uncal, "foot_strike_deg")) < 0.5


def test_unseen_far_side_foot_is_fine():
    from gaitlab.pose.skeleton import J

    cfg = PipelineConfig()
    seq, _ = synth_clip(GaitParams(near_side="r"))
    seq.vis[:, J["l_heel"]] = 0.0
    seq.vis[:, J["l_toe"]] = 0.0
    res = analyze_clip(seq, cfg)
    assert len(res.strides) > 0
    assert np.isfinite(med(res.strides, "cadence_spm"))


def test_knee_cycle_curve_resamples_ic_to_next_ic():
    from gaitlab.gait.features import analyze_clip, knee_cycle_curve, knee_flexion_deg

    seq, _ = synth_clip(duration_s=6.0)
    an = analyze_clip(seq, PipelineConfig())
    s = an.strides[1]
    curve = knee_cycle_curve(an.xy_up, seq.t, an.side, s.ic_frame, s.next_ic_frame)
    knee = knee_flexion_deg(an.xy_up, an.side)
    assert curve.shape == (101,)
    assert curve[0] == pytest.approx(knee[s.ic_frame])
    assert curve[-1] == pytest.approx(knee[s.next_ic_frame])
    assert curve.max() == pytest.approx(knee[s.ic_frame : s.next_ic_frame + 1].max(), abs=1.0)
