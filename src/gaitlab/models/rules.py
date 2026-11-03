"""Rules baseline (TECHNICAL_PLAN §7.1): grade = 5 − number of flagged tags, clipped to 1..5."""

from __future__ import annotations

from collections.abc import Mapping

from gaitlab.config import FeaturesConfig
from gaitlab.llm.verifier import flagged_tags


def rules_predict(
    window_row: Mapping[str, float | None], cfg: FeaturesConfig
) -> tuple[int, list[str]]:
    """Return (grade, tags). Raises ThresholdsNotCalibratedError while thresholds are null."""
    tags = flagged_tags(window_row, cfg)
    # backward_lean and forward_lean can't both hold for one window; no special-casing needed.
    return max(1, min(5, 5 - len(tags))), tags
