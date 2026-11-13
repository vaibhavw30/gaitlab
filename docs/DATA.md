# Data

Own treadmill video is the training data. Public datasets provide ground truth for validation, exact synthetic checks, and population norms. **Nothing under `data/` is committed.** Public datasets are downloaded by you, under their own terms, and never redistributed.

## 1. Directory layout (`data/`, git-ignored)

```
data/
├── sessions.csv                 # one row per clip (schema: TECHNICAL_PLAN §4.6)
├── raw/<session_id>/<clip_id>.mp4
├── interim/<session_id>/<clip_id>.<backend>.npz     # KeypointSeq
├── processed/<session_id>/{strides,windows}.parquet
├── processed/backend_pairs.<a>-<b>.parquet   # V7 matched strides (gaitlab validate own --other)
├── labels/labels.csv            # append-only
├── sft/{train,dev,test}.jsonl
├── splits.json                  # session_id → train|dev|test (frozen once written)
├── public/
│   ├── biocv/                   # as downloaded
│   ├── running_injury_clinic/
│   ├── fukuchi2017/
│   ├── cmu_mocap/
│   ├── guo2026/
│   └── stock/                   # Mixkit clips for smoke tests
├── manifest.json                # sha256 of every file used by a run → data_manifest_hash
└── synthetic/                   # gaitlab synth output; SYNTHETIC marker; never reported
    ├── windows.parquet, labels.csv, sessions.json
    ├── splits.json              # session split of the synthetic sessions (reset by synth)
    ├── predictions/<model>.csv  # baselines / predict --synthetic (+ .meta.json; LLM: .consistency.json)
    ├── sft/{train,dev,test}.jsonl
    ├── plots/                   # synthetic charts (never under docs/)
    ├── sessions.csv             # with --write-keypoints; path → the npz below
    ├── interim/<session_id>/<clip_id>.synthetic.npz
    └── processed/<session_id>/  # gaitlab process on synthetic keypoints
```

`session_id` = `YYYYMMDD-NN` (e.g. `20261105-01`). `clip_id` = `<condition>` or `<condition>-<n>`.

## 2. Capture protocol (print this)

Equipment: phone on a tripod, lens at hip height, perpendicular to the treadmill, 3–4 m away, whole body plus belt in frame, steady light, plain background if possible.

1. **Settings:** 120 fps minimum. Use 240 fps slow-motion for sessions where you will report contact time. Lock exposure and focus.
2. **Clothing:** fitted shorts and top; avoid loose shorts that hide the hip and knee.
3. **Standing clip (3 s):** stand still on the stopped belt, side-on, before running. This calibrates foot-strike angle.
4. **Warm up 6+ minutes off camera** at your normal pace. Treadmill kinematics shift for the first ~6 minutes.
5. **Conditions, about 60–75 s each (≥ 20 strides after settling), same speed:**
   - `natural`: normal running
   - `overstride`: reach the foot out ahead on purpose
   - `metronome`: metronome at +5–10% of natural cadence (count your natural cadence first)
   - `lean`: exaggerated forward lean from the waist, or exaggerated heel strike
   - `fatigue`: the last 60 s of a hard effort
6. **Log** the row in `sessions.csv` immediately: speed, incline, shoes, fps, camera height and distance.
7. **Ground truth:** for one 30-s stretch per session, hand-count steps in slow motion and note it in `notes` as `handcount_steps=<n>/30s`.

Target: 6–8 sessions → about 200 labeled 20-stride windows.

## 3. Public datasets

| Dataset | Use | Access / license | Loader |
|---|---|---|---|
| [BioCV (Univ. of Bath)](https://researchdata.bath.ac.uk/1258/), Evans et al., *Sci Data* 2024 | 15 people walking, running, jumping; synced multi-camera video, marker mocap and force plates. **Primary validation:** IC vs force plates, angles vs mocap | Read the archive's terms before download | `data/biocv.py` |
| [Running Injury Clinic Kinematic Dataset](https://figshare.com/articles/dataset/Running_Injury_Clinic_Kinematic_Dataset/24255795), Ferber et al. 2024 | 1,798 subjects, treadmill, 3D markers. Population norms and projected-2D feature checks | Figshare terms | `data/mocap.py` |
| [Fukuchi et al. 2017](https://peerj.com/articles/3298) | 28 runners at several speeds; 3D kinematics. Speed effects on features | CC-BY 4.0 | `data/mocap.py` |
| [CMU MoCap](https://paperswithcode.com/dataset/cmu-motion-capture) | Running trials for more projected sequences | Free for research and commercial use with credit; don't resell the data | `data/mocap.py` |
| [Guo et al. 2026 (Zenodo)](https://zenodo.org/records/18399031) | 21 people incl. running; OptiTrack 120 Hz + 2 phones 30 Hz; YOLOv8 2D (COCO-17, no feet) | CC BY 4.0. Check whether raw video is included | later (M5) |
| [AthletePose3D](https://arxiv.org/html/2503.07499v2) | Running and athletics video with mocap (~1.3M frames) | Check the terms on its GitHub | later (M5) |
| [Ole Miss stride-level dataset](https://egrove.olemiss.edu/hon_thesis/3469) | 73 treadmill videos, 4,600+ stride labels | Ask the author | if shared |
| [Mixkit stock clips](https://mixkit.co/free-stock-video/a-healthy-couple-running-on-a-treadmill-1321/) | Plumbing smoke tests only (24 fps) | Mixkit free license | none |
| [Sports2D demo](https://github.com/davidpagnon/Sports2D) | Angle reference outputs | BSD-3 | none |

### Projecting mocap to 2D (`data/mocap.py`)

1. Pick marker → canonical joint mapping per dataset (hip joint center from pelvis markers, knee from epicondyles, ankle from malleoli, heel and toe markers).
2. Find the direction of travel (principal horizontal axis of pelvis velocity, or the treadmill belt axis).
3. Orthographic projection onto the sagittal plane; scale to pixels as a 1080p frame would.
4. Resample to the target fps (120 and 240).
5. Optional noise model calibrated to the pose-model error you measure on BioCV.

Projected sequences are `backend="mocap_proj"`. Their features come with exact references (computed from the noise-free projection), which the tests compare against.

## 4. Splits

- Split by **session**, never by window. Windows within a session are near-duplicates.
- `data/splits.json` is written once when ≥ 6 sessions are labeled: the last 2 sessions by date → `test`, the one before → `dev`, the rest → `train`. Once written it is frozen; a CI-checked hash in `results/` records which split each result used.
- Baselines use leave-one-session-out CV on train + dev for model selection.

## 5. Synthetic data

`gaitlab synth` builds sessions from `data/synthetic.py` with random but known gait parameters per condition, plus rule-derived labels. All synthetic artifacts carry `synthetic=True` and land in `results/synthetic/`. They exist to dry-run the pipeline before your labels exist and are never reported.

With no video at all, the whole chain runs on synthetic keypoints ([ADR-0011](adr/0011-no-video-path-synthetic-keypoints-single-clip-overlay.md)):

```bash
uv run gaitlab synth --n-sessions 6 --write-keypoints
uv run gaitlab process --session 20260101-01 --sessions data/synthetic/sessions.csv --backend synthetic
uv run gaitlab process --clip data/synthetic/interim/20260101-01/natural.synthetic.npz
uv run gaitlab overlay --clip data/synthetic/interim/20260101-01/natural.synthetic.npz --gif
uv run gaitlab baselines --synthetic && uv run gaitlab build-sft --synthetic && uv run gaitlab eval --synthetic
# with the llm extra and gemma-3-270m-it in .hf_cache/ (ADR-0013); works on a Mac (MPS or CPU):
uv run gaitlab train --model 270m --sft-dir data/synthetic/sft --allow-synthetic --max-steps 20
uv run gaitlab predict --synthetic --model 270m && uv run gaitlab eval --synthetic
```

`eval --synthetic` writes `results/synthetic/grader/{ladder,tags,calibration}.json` ([ADR-0012](adr/0012-synthetic-grader-dry-run.md)), plus `verifier.json` and `consistency.json` once an LLM's predictions are present. The LoRA run goes to `runs/<run_id>/` with a `SYNTHETIC` marker.

Every output carries the flag: npz `backend="synthetic"` and `meta.synthetic`, a `synthetic` column in the CSV and parquet files, `SYNTHETIC` marker files, and a SYNTHETIC watermark on overlay frames.
