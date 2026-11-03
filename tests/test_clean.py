import numpy as np

from gaitlab.config import CleanCfg
from gaitlab.signal.clean import clean_keypoints, gate, interpolate_short_gaps, lowpass


def test_gate_sets_low_visibility_to_nan():
    xy = np.ones((3, 13, 2))
    vis = np.ones((3, 13))
    vis[1, 4] = 0.2
    out = gate(xy, vis, 0.5)
    assert np.isnan(out[1, 4]).all()
    assert np.isfinite(out[0]).all()


def test_interpolate_only_short_interior_gaps():
    s = np.arange(20, dtype=float)
    s[5:7] = np.nan  # short
    s[10:16] = np.nan  # long
    s[0] = np.nan  # edge
    out = interpolate_short_gaps(s, max_gap=3)
    np.testing.assert_allclose(out[5:7], [5, 6])
    assert np.isnan(out[10:16]).all()
    assert np.isnan(out[0])


def test_lowpass_keeps_slow_removes_fast():
    fps = 120.0
    t = np.arange(600) / fps
    slow = np.sin(2 * np.pi * 1.5 * t)
    fast = 0.5 * np.sin(2 * np.pi * 40 * t)
    out = lowpass(slow + fast, fps, 8.0)
    core = slice(60, -60)
    assert np.max(np.abs(out[core] - slow[core])) < 0.05


def test_lowpass_respects_nan_segments():
    s = np.sin(np.linspace(0, 20, 400))
    s[200:230] = np.nan
    out = lowpass(s, 120.0, 8.0)
    assert np.isnan(out[200:230]).all()
    assert np.isfinite(out[:200]).all() and np.isfinite(out[230:]).all()


def test_feature_cutoff_selected():
    xy = np.random.default_rng(0).normal(size=(300, 13, 2))
    vis = np.ones((300, 13))
    cfg = CleanCfg(lowpass_hz=5.0, feature_lowpass_hz=None)
    ev = clean_keypoints(xy, vis, 120.0, cfg)
    ft = clean_keypoints(xy, vis, 120.0, cfg, for_features=True)
    assert np.std(ev) < np.std(ft)
    np.testing.assert_allclose(ft, xy)
