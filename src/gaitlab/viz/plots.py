"""Charts (PRD visuals; TASKS M3). All plots read from results/*.json or processed parquet,
never from typed numbers.

Contract (matplotlib, saved as PNG at 2x and SVG under ``docs/assets/``):
- ``ladder_chart(results/grader/ladder.json, results/grader/intra_rater.json)``: QWK point + 95% CI
  whisker per model, dashed horizontal line at the intra-rater QWK, models in ladder order.
- ``knee_by_condition(strides.parquet)``: knee flexion across the gait cycle (0–100% of stride,
  resampled to 101 points), median line with IQR band per condition.
- ``stage_latency(results/bench/stages.json, results/bench/trt.json)``: stacked bars per stage,
  eager vs TensorRT.
- ``bland_altman(a, b, title)``, ``confusion(y_true, y_pred)``, ``reliability(probs, y_true)``.
Title each chart with the finding, not the topic, once the numbers exist.
"""

from __future__ import annotations
