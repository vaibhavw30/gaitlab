"""llm.train_lora: synthetic guard, run ids, assistant-only loss, 2-step CPU run (§7.2).

The model tests use gemma-3-270m-it from the repo-local cache and are marked ``slow``; anything
heavier than a couple of CPU steps belongs behind ``@pytest.mark.gpu``.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

from gaitlab.llm import runtime
from gaitlab.llm.train_lora import (
    check_synthetic,
    make_run_id,
    read_sft,
    strip_timing,
    train,
)

MODEL_270M = "google/gemma-3-270m-it"
has_llm = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft", "trl"))
requires_270m = pytest.mark.skipif(
    not has_llm or not runtime.is_cached(MODEL_270M),
    reason="needs the 'llm' extra and gemma-3-270m-it in .hf_cache/ (see README)",
)


def _record(i: int, synthetic: bool) -> dict:
    prompt = (
        "Grade this treadmill running window from 1 (poor) to 5 (excellent).\n"
        f"cadence_spm: {160 + i}.0 (iqr 2.0)\noverstride_L: 0.{20 + i} (iqr 0.010)"
    )
    target = f"grade: {1 + i % 5}\ntags: {'overstride' if i % 2 else 'none'}\ncue: Land closer."
    return {
        "window_id": f"2026010{1 + i % 3}-01:natural:{20 * i}",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "synthetic": synthetic,
    }


def _write_sft(d: Path, n_train: int = 8, n_dev: int = 4, synthetic: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for part, n in (("train", n_train), ("dev", n_dev), ("test", 2)):
        with open(d / f"{part}.jsonl", "w") as f:
            for i in range(n):
                f.write(json.dumps(_record(i, synthetic)) + "\n")
    return d


def test_synthetic_records_are_refused_unless_allowed(tmp_path: Path):
    recs = read_sft(_write_sft(tmp_path / "sft"))
    assert set(recs) == {"train", "dev"}, "test is never read by training"
    with pytest.raises(ValueError, match="allow_synthetic"):
        check_synthetic(recs, allow_synthetic=False)
    assert check_synthetic(recs, allow_synthetic=True) is True
    real = read_sft(_write_sft(tmp_path / "real", synthetic=False))
    assert check_synthetic(real, allow_synthetic=False) is False


def test_mixed_synthetic_and_real_records_are_always_refused(tmp_path: Path):
    recs = read_sft(_write_sft(tmp_path / "sft"))
    recs["dev"][0]["synthetic"] = False
    with pytest.raises(ValueError, match="mix"):
        check_synthetic(recs, allow_synthetic=True)


def test_train_refuses_synthetic_before_loading_anything(tmp_path: Path):
    with pytest.raises(ValueError, match="allow_synthetic"):
        train("270m", sft_dir=str(_write_sft(tmp_path / "sft")), out_root=str(tmp_path / "runs"))
    assert not (tmp_path / "runs").exists()


def test_run_id_format():
    now = dt.datetime(2026, 11, 4, 9, 5, 7, tzinfo=dt.UTC)
    assert make_run_id("270m", now, "sha256:abcdef0123456789") == "270m-20261104T090507Z-abcdef01"


def test_timing_is_stripped_from_trainer_state():
    state = {
        "global_step": 2,
        "best_metric": 1.5,
        "log_history": [
            {"loss": 2.0, "step": 1, "grad_norm": 3.0},
            {"eval_loss": 1.5, "eval_runtime": 0.4, "eval_samples_per_second": 9.0, "step": 2},
            {"train_runtime": 5.0, "train_steps_per_second": 0.4, "train_loss": 1.9, "step": 2},
        ],
    }
    out = strip_timing(state)
    keys = {k for e in out["log_history"] for k in e}
    assert not {k for k in keys if k.endswith(("_runtime", "_per_second"))}
    assert {"loss", "eval_loss", "train_loss", "grad_norm"} <= keys
    assert state["log_history"][1]["eval_runtime"] == 0.4, "input not mutated"


@pytest.mark.slow
@pytest.mark.own_process
@requires_270m
def test_only_assistant_tokens_carry_loss(tmp_path: Path):
    from gaitlab.config import load_train_lora_config
    from gaitlab.llm.train_lora import build_trainer

    recs = read_sft(_write_sft(tmp_path / "sft", n_train=4, n_dev=2))
    trainer = build_trainer("270m", recs, load_train_lora_config(), tmp_path / "run", device="cpu")
    batch = next(iter(trainer.get_train_dataloader()))
    tok = trainer.processing_class
    for ids, labels in zip(batch["input_ids"], batch["labels"], strict=True):
        trained = tok.decode(ids[labels != -100])
        ignored = tok.decode(ids[labels == -100], skip_special_tokens=True)
        assert trained.startswith("grade:"), trained
        assert "cue: Land closer." in trained
        assert "Grade this treadmill" not in trained and "cadence_spm" not in trained
        assert "Grade this treadmill" in ignored
        assert trained.rstrip().endswith("<end_of_turn>"), "the model must learn to stop"


@pytest.mark.slow
@pytest.mark.own_process
@requires_270m
def test_two_step_cpu_run_writes_a_flagged_run_dir(tmp_path: Path):
    run = train(
        "270m",
        sft_dir=str(_write_sft(tmp_path / "sft")),
        out_root=str(tmp_path / "runs"),
        allow_synthetic=True,
        max_steps=2,
        device="cpu",
    )
    assert run.parent == tmp_path / "runs" and run.name.startswith("270m-")
    meta = json.loads((run / "run.json").read_text())
    assert meta["synthetic"] is True and (run / "SYNTHETIC").exists()
    assert meta["model_id"] == MODEL_270M and meta["device"] == "cpu"
    assert meta["dtype"] == "float32" and meta["attn_implementation"] == "eager"
    assert meta["global_step"] == 2
    assert meta["config_hash"].startswith("sha256:")
    assert meta["data_manifest_hash"].startswith("sha256:")
    assert (run / "adapter" / "adapter_config.json").exists()
    assert json.loads((run / "config.json").read_text())["lora"]["r"] == 16
    state = json.loads((run / "trainer_state.json").read_text())
    assert not any(k.endswith(("_runtime", "_per_second")) for e in state["log_history"] for k in e)
    assert not (run / "checkpoints").exists()


@pytest.mark.gpu
@pytest.mark.own_process
@requires_270m
def test_four_step_cuda_run_in_bf16(tmp_path: Path):
    run = train(
        "270m",
        sft_dir=str(_write_sft(tmp_path / "sft")),
        out_root=str(tmp_path / "runs"),
        allow_synthetic=True,
        max_steps=4,
        device="cuda",
    )
    meta = json.loads((run / "run.json").read_text())
    assert (meta["device"], meta["dtype"], meta["global_step"]) == ("cuda", "bfloat16", 4)
