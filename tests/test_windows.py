from dataclasses import replace

from gaitlab.config import PipelineConfig, WindowsCfg
from gaitlab.data.synthetic import GaitParams, NoiseParams, synth_clip
from gaitlab.gait.features import analyze_clip
from gaitlab.gait.windows import make_windows
from gaitlab.types import FEATURE_FIELDS


def _strides(seconds=40.0):
    seq, _ = synth_clip(GaitParams(), duration_s=seconds, noise=NoiseParams(jitter_px=2, seed=0))
    return analyze_clip(seq, PipelineConfig()).strides


def test_non_overlapping_20_stride_windows():
    strides = _strides()
    ws = make_windows(strides, "20260101-01", "natural", WindowsCfg())
    assert len(ws) == len(strides) // 20
    assert all(w.n_strides == 20 for w in ws)
    assert ws[0].window_id == "20260101-01:natural:0"
    row = ws[0].flat()
    for f in FEATURE_FIELDS:
        assert f"{f}_med" in row and f"{f}_iqr" in row


def test_gap_starts_new_run():
    strides = _strides()
    broken = strides[:15] + strides[16:]  # drop one stride → no 20-run across the gap
    ws = make_windows(broken[:35], "s", "c", WindowsCfg())
    starts = [w.start_stride for w in ws]
    assert all(s >= 16 for s in starts)


def test_quality_filter():
    strides = [replace(s, quality=0.5) for s in _strides()]
    assert make_windows(strides, "s", "c", WindowsCfg(quality_min=0.8)) == []
