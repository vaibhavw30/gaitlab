# ADR-0011: No-video path: synthetic keypoint files, single-clip processing, canvas overlay

- Status: accepted
- Date: 2026-11-04

## Context
Until own recordings exist, the pipeline should run end to end without any video. `gaitlab synth` only wrote windows and labels; `process` needed a `data/sessions.csv` row and `data/interim` npz; `overlay` needed a video. The plan fixed the file schemas (TECHNICAL_PLAN §4) but not where synthetic keypoints live, how a lone clip is identified, or how outputs stay marked synthetic (CLAUDE.md rule 3).

## Decision
1. **Synthetic keypoints** go to `data/synthetic/interim/<session>/<clip>.synthetic.npz` (backend `synthetic`, `meta.synthetic = true`), with `data/synthetic/sessions.csv` in the §4.6 schema plus a `synthetic` column; `path` points at the npz. The loader ignores the extra column. The generator is split into `synthetic_session_clips` (keypoints) and `synthetic_sessions` (analysis + rule labels) with the same RNG order, so `synth` output is unchanged.
2. **`process --session`** reads `<sessions CSV dir>/interim/<session>/` and writes `<CSV dir>/processed/<session>/`, so the default CSV keeps `data/interim` → `data/processed` and the synthetic CSV stays inside `data/synthetic/`.
3. **`process --clip NPZ`** needs no sessions row: session id = the npz's folder name (`--session-id` overrides), clip id = file name up to the first dot, condition = the clip id's prefix if it is a known condition (`--condition` overrides), speed unknown unless `--speed-kmh`. Outputs are `<clip>.strides.parquet` / `<clip>.windows.parquet` (a prefix, so they never overwrite or get picked up as a session's `windows.parquet`) in `data/processed/<session>/`, or `data/synthetic/processed/<session>/` for synthetic keypoints.
4. **Synthetic flag on outputs:** strides and windows rows get a `synthetic` bool column from `KeypointSeq.synthetic`; synthetic output folders get a `SYNTHETIC` marker; a session mixing synthetic and real keypoints is rejected. Overlay frames of synthetic sequences carry a SYNTHETIC watermark and the file's comment metadata says so.
5. **Overlay drawing:** the skeleton is the stored keypoints (what the backend produced); the angle text uses the feature-filtered signal (`ClipAnalysis.xy_up`), so the number shown on an IC frame equals that stride's feature. IC/TO markers start on the event frame and are held for 0.1 s. Without `--video` the background is a plain canvas of `KeypointSeq.size`. mp4 frames keep `seq.t` as PTS (VFR timing survives); GIFs are sampled at 15 fps by timestamp.

## Consequences
- `windows.parquet` and `strides.parquet` gain a `synthetic` column (TECHNICAL_PLAN §4.3–4.4). Readers that select `<feature>_med/_iqr` columns are unaffected.
- A lone clip in a folder not named like a session gets that folder name as its session id. Pass `--session-id` for anything that will be split or labeled.
- The synthetic runner's default scale (600 px/m on 1080p) puts the head above the canvas top. Fine for checking legs and events; change `synth_clip(px_per_m=...)` if a full-body synthetic GIF is wanted.
