"""Fail if README.md states result-like numbers outside the generated RESULTS block (ADR-0007).

Flags percentages, multipliers ("3x", "2.1×"), and metric names followed by a number
(kappa, QWK, MAE, RMSE, F1, accuracy, agreement, speedup, latency, fps achieved).
Design parameters ("20-stride", "120 fps capture", "Gemma 3 1B") are not flagged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PATTERNS = [
    re.compile(r"\b\d+(\.\d+)?\s?%"),
    re.compile(r"\b\d+(\.\d+)?\s?[x×]\b(?!\d)"),
    re.compile(
        r"\b(kappa|qwk|mae|rmse|f1|accuracy|agreement|speedup|latency|throughput)\b[^\n|]{0,25}?\d",
        re.I,
    ),
]


def check(path: str = "README.md") -> list[str]:
    text = Path(path).read_text()
    outside = re.sub(re.escape(START) + r".*?" + re.escape(END), "", text, flags=re.S)
    problems = []
    for lineno, line in enumerate(outside.splitlines(), 1):
        if line.strip().startswith(("http", "[!")):
            continue
        for pat in PATTERNS:
            if pat.search(line):
                problems.append(f"{path}:{lineno}: {line.strip()}")
                break
    return problems


if __name__ == "__main__":
    probs = check(sys.argv[1] if len(sys.argv) > 1 else "README.md")
    for p in probs:
        print("unsourced number:", p)
    sys.exit(1 if probs else 0)
