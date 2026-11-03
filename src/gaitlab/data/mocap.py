"""Project 3D marker data to a 2D side view (DATA.md §3, TASKS M0, EVALUATION V3).

Contract:
- ``load_fukuchi_trial(path) -> MarkerTrial`` and ``load_ric_trial(path) -> MarkerTrial``: marker
  name → (T, 3) array in metres, plus sample rate. Fukuchi 2017 first (CC-BY, small).
- ``marker_map_<dataset>``: marker(s) → canonical joint. Hip = hip joint centre estimated from
  pelvis markers (or greater trochanter if that's all there is); knee = mid-epicondyles;
  ankle = mid-malleoli; heel = calcaneus marker; toe = 2nd/1st metatarsal head or toe marker;
  shoulder = acromion.
- ``project_sagittal(trial, marker_map, fps_out, px_per_m=600, size=(1920, 1080)) -> KeypointSeq``:
  find the forward axis (principal horizontal axis of pelvis velocity, or the treadmill axis
  if documented), orthographic-project onto (forward, vertical), resample to ``fps_out`` with
  linear interpolation, convert to y-down pixels with ground ~100 px above the bottom, vis = 1.
  backend = "mocap_proj".
- ``reference_events(trial) -> dict[side, (ic, to)]`` from force plates when available, else
  from heel/toe marker kinematics at the native rate (document which in the result notes).
Never commit dataset files. Tests on real files are ``@pytest.mark.data``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaitlab.types import KeypointSeq


@dataclass
class MarkerTrial:
    markers: dict[str, np.ndarray]  # name → (T, 3) metres
    rate_hz: float
    meta: dict


def project_sagittal(
    trial: MarkerTrial,
    marker_map: dict[str, list[str]],
    fps_out: float,
    px_per_m: float = 600.0,
    size: tuple[int, int] = (1920, 1080),
) -> KeypointSeq:
    raise NotImplementedError("TASKS.md M0: mocap projection")
