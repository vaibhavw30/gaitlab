"""eval.run_eval.run: predictions + labels + split → ladder, tags, calibration (G1, G3, G5)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gaitlab.eval.run_eval import PRED_COLUMNS, run
from gaitlab.report.results import load_reportable

SPLIT = {"20260101-01": "train", "20260108-01": "dev", "20260115-01": "test", "20260122-01": "test"}
VOCAB = ["overstride", "low_cadence"]


def _labels(path: Path, rater: str = "synthetic") -> dict[str, tuple[int, str]]:
    rng = np.random.default_rng(0)
    truth: dict[str, tuple[int, str]] = {}
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, ["window_id", "grade", "tags", "cue", "rater", "mode", "labeled_at"])
        wr.writeheader()
        for sid in SPLIT:
            for k in range(6):
                wid = f"{sid}:natural:{20 * k}"
                g = int(rng.integers(2, 6))
                tags = "overstride" if g <= 3 else ""
                truth[wid] = (g, tags)
                wr.writerow(
                    {"window_id": wid, "grade": g, "tags": tags, "cue": "", "rater": rater,
                     "mode": "normal", "labeled_at": "2026-01-01T12:00:00"}
                )  # fmt: skip
    return truth


def _write_preds(d: Path, name: str, truth: dict, probs: bool, noise: int) -> None:
    rng = np.random.default_rng(noise)
    with open(d / f"{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, PRED_COLUMNS)
        wr.writeheader()
        for wid, (g, tags) in truth.items():
            if SPLIT[wid.split(":")[0]] != "test":
                continue
            pred = int(np.clip(g + rng.integers(-noise, noise + 1), 1, 5)) if noise else g
            row = {"window_id": wid, "pred_grade": pred, "tags": tags, "cue": "", "synthetic": True}
            p = np.full(5, 0.05)
            p[pred - 1] = 0.8
            for k in range(5):
                row[f"p{k + 1}"] = float(p[k]) if probs else ""
            wr.writerow(row)


@pytest.fixture
def inputs(tmp_path: Path):
    labels = tmp_path / "labels.csv"
    truth = _labels(labels)
    preds = tmp_path / "predictions"
    preds.mkdir()
    _write_preds(preds, "rules", truth, probs=False, noise=0)
    _write_preds(preds, "logreg", truth, probs=True, noise=1)
    _write_preds(preds, "lgbm", truth, probs=True, noise=1)
    (preds / "lgbm.meta.json").write_text(json.dumps({"params": {"num_leaves": 7}}))
    split = tmp_path / "splits.json"
    split.write_text(json.dumps(SPLIT))
    return tmp_path, preds, labels, split


def test_run_writes_synthetic_ladder_tags_and_calibration(inputs):
    root, preds, labels, split = inputs
    out = run(
        str(preds), str(labels), str(split), synthetic=True, tag_vocab=VOCAB, n_boot=50,
        results_root=str(root / "results"),
    )  # fmt: skip
    assert set(out) == {"ladder", "tags", "calibration"}
    for name, path in out.items():
        assert path == root / "results" / "synthetic" / "grader" / f"{name}.json"
        d = json.loads(path.read_text())
        assert d["synthetic"] is True and d["area"] == "grader"
    ladder = json.loads(out["ladder"].read_text())["metrics"]
    assert ladder["order"] == ["rules", "logreg", "lgbm"]
    assert ladder["split"] == "test"
    n_test = sum(v == "test" for v in SPLIT.values()) * 6
    for m in ladder["order"]:
        assert ladder["models"][m]["n_windows"] == n_test
        assert ladder["models"][m]["n_sessions"] == 2
    assert ladder["models"]["lgbm"]["selection"] == {"params": {"num_leaves": 7}}
    assert ladder["best"]["model"] == "rules"  # rules copy the labels here
    assert isinstance(ladder["best"]["qwk"], float)
    tags = json.loads(out["tags"].read_text())["metrics"]
    assert set(tags["models"]) == {"rules", "logreg", "lgbm"}
    assert "low_cadence" in tags["models"]["rules"]["absent_in_truth"]
    cal = json.loads(out["calibration"].read_text())["metrics"]
    assert set(cal["models"]) == {"logreg", "lgbm"}
    assert cal["no_probabilities"] == ["rules"]
    bins = cal["models"]["logreg"]["reliability"]
    assert sum(b["count"] for b in bins) == n_test
    # Synthetic results never reach the README.
    assert load_reportable(root / "results") == []


def test_run_refuses_to_report_synthetic_labels_as_real(inputs):
    root, preds, labels, split = inputs
    with pytest.raises(ValueError, match="synthetic"):
        run(str(preds), str(labels), str(split), synthetic=False, tag_vocab=VOCAB, n_boot=10,
            results_root=str(root / "results"))  # fmt: skip


def test_run_requires_every_model_to_cover_the_same_test_windows(inputs):
    root, preds, labels, split = inputs
    lines = (preds / "lgbm.csv").read_text().splitlines()
    (preds / "lgbm.csv").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="lgbm"):
        run(str(preds), str(labels), str(split), synthetic=True, tag_vocab=VOCAB, n_boot=10,
            results_root=str(root / "results"))  # fmt: skip


# --- LLM predictions: verifier (G4) and consistency (G6) -------------------------------------

CFG_TAGS = ["overstride", "low_cadence"]


def _llm_inputs(root: Path, preds: Path, truth: dict, cfg) -> dict[str, dict]:
    """An LLM prediction file whose cues alternate grounded / ungrounded, plus its windows."""
    over = cfg.tags["overstride"].threshold
    windows, rows = {}, []
    test = [w for w in truth if SPLIT[w.split(":")[0]] == "test"]
    for k, wid in enumerate(test):
        windows[wid] = {"window_id": wid, "overstride_L_med": over + 0.1, "cadence_spm_med": 170.0}
        cue = "Land closer under your hips." if k % 2 == 0 else "Great job."
        row = {"window_id": wid, "pred_grade": truth[wid][0], "tags": "overstride", "cue": cue,
               "synthetic": True}  # fmt: skip
        for g in range(1, 6):
            row[f"p{g}"] = 0.6 if g == truth[wid][0] else 0.1
        rows.append(row)
    with open(preds / "gemma3_270m_lora.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, PRED_COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    (preds / "gemma3_270m_lora.meta.json").write_text(
        json.dumps({"kind": "llm", "generates_cues": True, "grade_token_form": "bare_digit"})
    )
    runs = {
        wid: {
            "grades": [truth[wid][0]] * 3 if k % 3 else [truth[wid][0], 1, truth[wid][0]],
            "tags": [["overstride"]] * 3,
            "cues": ["Land closer."] * 3 if k % 2 == 0 else ["a", "b", "a"],
        }
        for k, wid in enumerate(test)
    }
    (preds / "gemma3_270m_lora.consistency.json").write_text(
        json.dumps({"runs": 3, "temperature": 0.7, "windows": runs})
    )
    return windows


def test_llm_joins_the_ladder_with_verifier_and_consistency(inputs, synthetic_features_cfg):
    root, preds, labels, split = inputs
    truth = {w: (g, t) for w, (g, t) in _labels(root / "labels2.csv").items()}
    windows = _llm_inputs(root, preds, truth, synthetic_features_cfg)
    out = run(
        str(preds), str(labels), str(split), synthetic=True, tag_vocab=VOCAB, n_boot=50,
        results_root=str(root / "results"), windows=windows, features_cfg=synthetic_features_cfg,
    )  # fmt: skip
    assert set(out) == {"ladder", "tags", "calibration", "verifier", "consistency"}
    for name in ("verifier", "consistency"):
        assert out[name] == root / "results" / "synthetic" / "grader" / f"{name}.json"
        assert json.loads(out[name].read_text())["synthetic"] is True
    ladder = json.loads(out["ladder"].read_text())["metrics"]
    assert ladder["order"][-1] == "gemma3_270m_lora"
    assert ladder["models"]["gemma3_270m_lora"]["selection"]["grade_token_form"] == "bare_digit"
    cal = json.loads(out["calibration"].read_text())["metrics"]
    assert "gemma3_270m_lora" in cal["models"]

    ver = json.loads(out["verifier"].read_text())["metrics"]
    assert set(ver["models"]) == {"gemma3_270m_lora"}
    assert ver["no_cues"] == ["rules", "logreg", "lgbm"]
    v = ver["models"]["gemma3_270m_lora"]
    n_test = len(windows)
    assert v["n_windows"] == n_test and v["n_passed"] == (n_test + 1) // 2
    assert v["failures"] == {"cue_misses_tags": n_test // 2}
    assert 0.0 <= v["pass_rate"]["low"] <= v["pass_rate"]["point"] <= v["pass_rate"]["high"]

    con = json.loads(out["consistency"].read_text())["metrics"]
    c = con["models"]["gemma3_270m_lora"]
    assert c["runs"] == 3 and c["temperature"] == 0.7
    n_unstable = sum(1 for k in range(n_test) if k % 3 == 0)
    assert c["grade_identical"]["point"] == pytest.approx(1 - n_unstable / n_test)
    assert c["tags_identical_sampled"]["point"] == 1.0
    assert c["cue_identical_sampled"]["point"] == pytest.approx(((n_test + 1) // 2) / n_test)
    assert load_reportable(root / "results") == []


def test_llm_predictions_need_windows_for_the_verifier(inputs, synthetic_features_cfg):
    root, preds, labels, split = inputs
    truth = _labels(root / "labels2.csv")
    _llm_inputs(root, preds, truth, synthetic_features_cfg)
    with pytest.raises(ValueError, match="windows"):
        run(str(preds), str(labels), str(split), synthetic=True, tag_vocab=VOCAB, n_boot=10,
            results_root=str(root / "results"))  # fmt: skip


def test_consistency_file_must_cover_the_test_windows(inputs, synthetic_features_cfg):
    root, preds, labels, split = inputs
    truth = _labels(root / "labels2.csv")
    windows = _llm_inputs(root, preds, truth, synthetic_features_cfg)
    p = preds / "gemma3_270m_lora.consistency.json"
    d = json.loads(p.read_text())
    d["windows"].pop(next(iter(d["windows"])))
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="consistency"):
        run(str(preds), str(labels), str(split), synthetic=True, tag_vocab=VOCAB, n_boot=10,
            results_root=str(root / "results"), windows=windows,
            features_cfg=synthetic_features_cfg)  # fmt: skip
