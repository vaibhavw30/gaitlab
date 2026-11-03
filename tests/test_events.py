import numpy as np
import pytest

from gaitlab.config import PipelineConfig
from gaitlab.data.synthetic import CONDITIONS, GaitParams, NoiseParams, condition_params, synth_clip
from gaitlab.gait import geometry
from gaitlab.gait.events import detect_events, stride_pairs
from gaitlab.signal.clean import clean_keypoints


def _detect(seq, cfg):
    xy = geometry.to_y_up(clean_keypoints(seq.xy, seq.vis, seq.fps, cfg.clean), seq.size[1])
    side = geometry.near_side(seq.vis)
    d = geometry.facing_direction(xy, side)
    return side, detect_events(xy, side, d, geometry.leg_length(xy, side), seq.fps, cfg.events)


def _match(detected: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.array([int(detected[np.argmin(np.abs(detected - x))] - x) for x in truth])


@pytest.mark.parametrize("direction", [1, -1])
def test_clean_events_within_one_frame(direction):
    cfg = PipelineConfig()
    cfg.clean.lowpass_hz = None
    seq, truth = synth_clip(GaitParams(direction=direction))
    side, ev = _detect(seq, cfg)
    assert len(ev.ic) == len(truth.ic[side])
    assert np.abs(_match(ev.ic, truth.ic[side])).max() <= 1
    assert np.abs(_match(ev.to, truth.to[side])).max() <= 1


@pytest.mark.parametrize("condition", list(CONDITIONS))
@pytest.mark.parametrize("seed", [0, 1])
def test_noisy_events_default_config(condition, seed):
    cfg = PipelineConfig()
    p = condition_params(GaitParams(direction=1 if seed else -1), condition)
    seq, truth = synth_clip(p, noise=NoiseParams(jitter_px=3.0, dropout_frac=0.01, seed=seed))
    side, ev = _detect(seq, cfg)
    assert len(ev.ic) == len(truth.ic[side]), "missed or extra contacts"
    assert np.abs(_match(ev.ic, truth.ic[side])).max() <= 2
    assert np.abs(_match(ev.to, truth.to[side])).max() <= 2


def test_stride_pairs_respect_time_bounds():
    cfg = PipelineConfig()
    seq, _truth = synth_clip(GaitParams())
    _, ev = _detect(seq, cfg)
    pairs = stride_pairs(ev, seq.t, cfg.events)
    assert len(pairs) == len(ev.ic) - 1
    cfg.events.stride_time_max_s = 0.5  # shorter than this runner's stride
    assert stride_pairs(ev, seq.t, cfg.events) == []
