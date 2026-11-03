"""Population percentiles from projected Running Injury Clinic data (FR-16, TASKS M5).

Contract:
``build_norms(projected_dir, cfg) -> pandas.DataFrame``: run ``analyze_clip`` on each projected
running trial (healthy subjects only, per dataset metadata), summarize per subject (median per
feature), and store percentiles (5..95 by 5) per feature, overall and by speed band, to
``data/processed/norms.parquet``. Also used to propose (not set) tag thresholds in M2 (ADR-0009).
``percentile_of(feature, value, norms, speed_kmh=None) -> float``.
"""

from __future__ import annotations
