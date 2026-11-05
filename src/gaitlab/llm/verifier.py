"""Cue verifier (TECHNICAL_PLAN §7.3): a cue must point at a feature that is actually off.

Also hosts tag evaluation shared with ``models.rules``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from gaitlab.config import FeaturesConfig


class ThresholdsNotCalibratedError(RuntimeError):
    """A needed tag threshold in configs/features.yaml is still null (CLAUDE.md rule 5)."""


# Short alias used in docs.
ThresholdsNotCalibrated = ThresholdsNotCalibratedError


def tag_holds(tag: str, window_row: Mapping[str, float | None], cfg: FeaturesConfig) -> bool:
    """Whether ``tag``'s condition holds for this window. NaN/missing features → False."""
    rule = cfg.tags[tag]
    if rule.threshold is None:
        raise ThresholdsNotCalibratedError(f"threshold for tag {tag!r} is null")
    v = window_row.get(rule.feature)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return v > rule.threshold if rule.op == ">" else v < rule.threshold


def flagged_tags(window_row: Mapping[str, float | None], cfg: FeaturesConfig) -> list[str]:
    """All tags whose condition holds, in vocabulary order."""
    return [t for t in cfg.tags if tag_holds(t, window_row, cfg)]


FAILURE_CODES = ("unknown_tag", "unsupported_tag", "cue_misses_tags", "cue_implies_untagged")


@dataclass
class VerifierResult:
    """``codes[i]`` (one of ``FAILURE_CODES``) classifies ``reasons[i]`` for the G4 histogram."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)


def verify(
    window_row: Mapping[str, float | None],
    tags: Sequence[str],
    cue: str,
    cfg: FeaturesConfig,
) -> VerifierResult:
    """Pass iff every tag is known, every tag's condition holds, and the cue mentions a keyword
    of at least one generated tag. A window with no generated tags passes only with an empty
    cue or a cue that names no tag keywords (nothing to correct)."""
    reasons: list[str] = []
    codes: list[str] = []
    unknown = [t for t in tags if t not in cfg.tags]
    if unknown:
        reasons.append(f"unknown tags: {unknown}")
        codes.append("unknown_tag")
    known = [t for t in tags if t in cfg.tags]
    unsupported = [t for t in known if not tag_holds(t, window_row, cfg)]
    if unsupported:
        reasons.append(f"tags not supported by features: {unsupported}")
        codes.append("unsupported_tag")
    cue_l = cue.lower()
    if known:
        if not any(kw.lower() in cue_l for t in known for kw in cfg.tags[t].keywords):
            reasons.append("cue mentions none of the generated tags")
            codes.append("cue_misses_tags")
    else:
        mentioned = [t for t, r in cfg.tags.items() if any(k.lower() in cue_l for k in r.keywords)]
        if mentioned:
            reasons.append(f"cue implies issues {mentioned} but no tags were generated")
            codes.append("cue_implies_untagged")
    return VerifierResult(passed=not reasons, reasons=reasons, codes=codes)
