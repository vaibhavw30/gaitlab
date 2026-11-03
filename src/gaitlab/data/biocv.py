"""BioCV dataset loader (Evans et al., Sci Data 2024). TASKS M0, EVALUATION V1/V2.

**Confirm the dataset terms with the user before downloading.** Never commit its files.

Contract:
- ``list_running_trials(root) -> list[BioCVTrial]``: participant, trial name, the video path of
  the camera most perpendicular to the running direction (inspect calibration files to choose),
  mocap file, force-plate file, video fps.
- ``force_plate_events(trial, threshold_n=20.0) -> dict[side, (ic_times_s, to_times_s)]``:
  vertical GRF crossing the threshold up (IC) and down (TO). Overground running gives only
  a few contacts per trial; keep that in mind for sample sizes.
- ``align_to_video(events_s, video_t)``: map times to nearest video frames using the dataset's
  sync information; report the sync uncertainty in the result notes.
Result writers: ``results/validation/biocv_events.json`` (IC/TO error in frames and ms; % within
±1 frame; missed or extra events) and ``biocv_angles.json`` (Bland–Altman vs mocap angles).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BioCVTrial:
    participant: str
    trial: str
    video_path: str
    mocap_path: str
    forceplate_path: str
    fps: float


def list_running_trials(root: str) -> list[BioCVTrial]:
    raise NotImplementedError("TASKS.md M0: BioCV loader (confirm license first)")
