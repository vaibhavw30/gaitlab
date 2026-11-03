from collections import Counter

import numpy as np

from gaitlab.config import PipelineConfig
from gaitlab.data.synthetic import synthetic_sessions
from gaitlab.eval.run_eval import grade_metrics
from gaitlab.models.rules import rules_predict


def test_synthetic_sessions_dry_run(synthetic_features_cfg):
    windows, labels, sessions = synthetic_sessions(3, PipelineConfig(), synthetic_features_cfg)
    assert len(windows) == len(labels) > 0
    assert all(s["synthetic"] for s in sessions)
    assert {lab["rater"] for lab in labels} == {"synthetic"}
    # Conditions should produce a spread of grades, not one cluster.
    assert len(Counter(lab["grade"] for lab in labels)) >= 2
    # Plumbing check: the rules baseline reproduces its own labels, so agreement is perfect.
    preds = [rules_predict(w, synthetic_features_cfg)[0] for w in windows]
    groups = [w["session_id"] for w in windows]
    res = grade_metrics([lab["grade"] for lab in labels], preds, groups, n_boot=50)
    assert np.isclose(res["exact"]["point"], 1.0)
