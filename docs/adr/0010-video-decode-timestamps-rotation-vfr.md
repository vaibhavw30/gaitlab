# ADR-0010: Video decode: timestamp origin, rotation, VFR flag, frame count

- Status: accepted
- Date: 2026-11-03

## Context
TECHNICAL_PLAN §4.2 and the `io/video.py` contract fix the essentials: timing from container PTS, RGB frames in presentation order, rotation metadata applied, a `vfr` flag when PTS deltas vary by more than 1 %, and a `LowFpsError` below `capture.min_fps`. Four details were left open:

1. **What "seconds from clip start" means.** Streams can start at a non-zero PTS (edit lists, B-frame delay).
2. **Where rotation comes from.** Phones record landscape sensor frames with a display matrix (the old `rotate=90` tag). PyAV 18 exposes this only on decoded frames (`VideoFrame.rotation`, counter-clockwise degrees, FFmpeg convention), not in stream metadata.
3. **Rounding vs VFR.** A plain "deltas vary by more than 1 %" rule flags constant-rate clips stored in coarse time bases as VFR. Matroska stores milliseconds, so 120 fps gives deltas of 8 and 9 ms. `tests/test_video.py::test_coarse_timebase_rounding_is_not_vfr` reproduces this.
4. **Frame count.** `stream.frames` is 0 for some containers (Matroska), so the scaffold's contract allowed an estimate.

## Decision
1. `t` is measured from the **first presented frame's PTS**: `t = (pts − pts₀) × time_base`, so `t[0] == 0`. Event and feature math only uses differences, and `overlay` re-decodes through the same function, so frame alignment is consistent.
2. Rotation is read per frame from `VideoFrame.rotation`, rounded to the nearest multiple of 90°, and applied with `np.rot90` (counter-clockwise positive). Output is C-contiguous for the pose backends. `probe` decodes one frame to learn the displayed size. Display-matrix flips are ignored, since phone camera video doesn't use them.
3. `vfr` is True when any PTS delta deviates from the median delta by more than `max(1 % × median, 1 tick)`. The one-tick allowance absorbs time-base rounding only.
4. `n_frames` is **exact**: `probe` demuxes every packet without decoding and counts those with a PTS. That pass also supplies the PTS for the VFR check.

## Consequences
- `probe` reads the whole file once (demux only, no decode). That's cheap next to pose estimation.
- Jitter of one tick is not reported as VFR even when one tick exceeds 1 % of the period. Downstream timing uses the `t` array regardless (TECHNICAL_PLAN §5.3), so the flag is informational.
- Videos with mirrored display matrices would come out unflipped. If one shows up, add the flip and a test.
- The PyAV tests are skipped when the `video` extra isn't installed. CI installs base + `ml` + `dev` only (TECHNICAL_PLAN §11), so CI doesn't exercise them unless `--extra video` is added to `ci.yml`.
