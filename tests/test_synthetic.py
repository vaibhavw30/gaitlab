import numpy as np
import pytest

from gaitlab.data.synthetic import CONDITIONS, GaitParams, condition_params, synth_clip
from gaitlab.gait import geometry
from gaitlab.pose.skeleton import J


def test_stance_fraction_matches_duty_factor():
    p = GaitParams()
    _, truth = synth_clip(p, duration_s=20)
    frac = truth.stance["r"].mean()
    assert abs(frac - p.duty_factor) < 0.01


def test_foot_on_ground_during_stance_only():
    seq, truth = synth_clip(GaitParams(), duration_s=10)
    up = geometry.to_y_up(truth.xy_clean, seq.size[1])
    ground_px = 100.0
    low = np.minimum(up[:, J["r_heel"], 1], up[:, J["r_toe"], 1]) - ground_px
    st = truth.stance["r"]
    assert np.allclose(low[st], 0.0, atol=1e-6)
    assert (low[~st] > 0.02 * truth.leg_length_px).all()  # clear of the contact threshold


def test_leg_length_exact_and_ic_count():
    p = GaitParams()
    seq, truth = synth_clip(p, duration_s=12)
    up = geometry.to_y_up(truth.xy_clean, seq.size[1])
    assert abs(geometry.leg_length(up, "r") - truth.leg_length_px) < 1e-3
    expected = 12 / p.stride_time_s
    assert abs(len(truth.ic["r"]) - expected) <= 2


def test_direction_flips_image_x():
    a, _ = synth_clip(GaitParams(direction=1), duration_s=2)
    b, _ = synth_clip(GaitParams(direction=-1), duration_s=2)
    up_a = geometry.to_y_up(a.xy, a.size[1])
    up_b = geometry.to_y_up(b.xy, b.size[1])
    assert geometry.facing_direction(up_a, "r") == 1
    assert geometry.facing_direction(up_b, "r") == -1


def test_near_side_from_visibility():
    seq, _ = synth_clip(GaitParams(near_side="l"), duration_s=2)
    assert geometry.near_side(seq.vis) == "l"


@pytest.mark.parametrize("condition", [*CONDITIONS, "standing"])
def test_default_frame_holds_the_whole_body(condition):
    """Every joint, head included, lies inside the default 1920x1080 frame."""
    if condition == "standing":
        seq, truth = synth_clip(standing=True, duration_s=1.0)
    else:
        seq, truth = synth_clip(condition_params(GaitParams(), condition), duration_s=4.0)
    w, h = seq.size
    x, y = truth.xy_clean[..., 0], truth.xy_clean[..., 1]
    assert x.min() >= 0 and x.max() < w
    assert y.min() >= 0 and y.max() < h, f"y range {y.min():.0f}..{y.max():.0f} px"
