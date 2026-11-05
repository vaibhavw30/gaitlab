"""LoRA fine-tuning (TECHNICAL_PLAN §7.2, configs/train_lora.yaml). TASKS M3.

Contract:
``train(model_key: Literal["1b", "270m"], sft_dir="data/sft", cfg_path="configs/train_lora.yaml",
        out_root="runs", allow_synthetic=False, *, max_steps=None, device=None) -> Path``
- Read train/dev JSONL (``test`` is never read here) into in-memory ``datasets.Dataset``s, so
  nothing is cached outside the repo. Refuse to run if any record has ``synthetic=True`` unless
  ``allow_synthetic=True`` (dry runs), and always refuse a mix of synthetic and real records.
- The grade tokens are checked (``llm.grade.grade_tokens_for``) before the model is loaded.
- Device cuda > mps > cpu (``llm.runtime``). CUDA trains in bf16 when ``train.bf16``; MPS and CPU
  load float32. Attention is ``eager`` everywhere (recommended for Gemma 3 training).
- ``peft.LoraConfig`` from the YAML; ``trl.SFTTrainer`` on the chat-format ``messages`` field
  with ``assistant_only_loss=True``. Gemma 3's chat template has no ``{% generation %}`` markers;
  TRL swaps in its Gemma 3 training template, so only the assistant turn (including its
  ``<end_of_turn>``) carries loss. ``tests/test_train_lora.py`` checks this on one batch.
- Seeds fixed; eval on dev each epoch; the best checkpoint by dev loss is kept.
  ``max_steps`` (smoke runs) caps optimizer steps and overrides ``train.epochs``.
- Write ``runs/<run_id>/`` with: ``adapter/`` (PEFT weights), ``config.json`` (resolved
  config), ``run.json`` (config hash, train/dev manifest hash, device, dtype, versions,
  ``synthetic``), ``trainer_state.json`` and, for dry runs, a ``SYNTHETIC`` marker.
  ``run_id`` = ``<model_key>-<UTC timestamp>-<short config hash>``. Intermediate checkpoints
  are deleted after the best adapter is saved.
- No timing is recorded: ``*_runtime`` and ``*_per_second`` entries are stripped from the
  trainer state (speed is measured only by ``bench/``, on the GPU host).
- Optional W&B logging if ``WANDB_API_KEY`` is set and the device is CUDA; never required.
GPU smoke test (``@pytest.mark.gpu``): 4 optimizer steps on 8 synthetic records with 270m.
CPU variant (``@pytest.mark.slow``): 2 steps.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SYNTHETIC_MARKER = "Trained on synthetic records. Never report anything produced from this run.\n"
TIMING_SUFFIXES = ("_runtime", "_per_second")


def read_sft(sft_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """train/dev JSONL → records. Each record needs ``messages``."""
    out: dict[str, list[dict[str, Any]]] = {}
    for part in ("train", "dev"):
        path = Path(sft_dir) / f"{part}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run gaitlab build-sft first")
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not recs:
            raise ValueError(f"{path} is empty")
        out[part] = recs
    return out


def check_synthetic(
    records: Mapping[str, Sequence[Mapping[str, Any]]], allow_synthetic: bool
) -> bool:
    """Return whether the records are synthetic; refuse mixes and unallowed synthetic data."""
    flags = {bool(r.get("synthetic", False)) for recs in records.values() for r in recs}
    if len(flags) > 1:
        raise ValueError("SFT records mix synthetic and real windows; rebuild them separately")
    synthetic = flags == {True}
    if synthetic and not allow_synthetic:
        raise ValueError(
            "SFT records are synthetic; pass allow_synthetic=True (--allow-synthetic) for a dry run"
        )
    return synthetic


def make_run_id(model_key: str, now: dt.datetime, cfg_hash: str) -> str:
    short = cfg_hash.split(":", 1)[-1][:8]
    return f"{model_key}-{now.astimezone(dt.UTC):%Y%m%dT%H%M%SZ}-{short}"


def strip_timing(state: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of a trainer state without wall-clock entries (runtime, per-second rates)."""
    out = dict(state)
    out["log_history"] = [
        {k: v for k, v in e.items() if not k.endswith(TIMING_SUFFIXES)}
        for e in state.get("log_history", [])
    ]
    return out


def build_trainer(
    model_key: str,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    cfg,
    out_dir: str | Path,
    *,
    device: str | None = None,
    max_steps: int | None = None,
):
    """An ``SFTTrainer`` ready to ``train()``; checkpoints go to ``<out_dir>/checkpoints``."""
    from gaitlab.llm import runtime
    from gaitlab.llm.grade import grade_tokens_for, load_tokenizer

    model_id = cfg.models[model_key]
    tokenizer = load_tokenizer(model_id)
    grade_tokens_for(tokenizer, cfg.grade.grade_tokens)  # before any weights load
    tokenizer.padding_side = "right"

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    device = runtime.pick_device(device)
    settings = runtime.load_settings(device, training=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=runtime.torch_dtype(settings.dtype),
        attn_implementation=settings.attn_implementation,
        **runtime.from_pretrained_kwargs(model_id),
    )  # fmt: skip
    t = cfg.train
    use_wandb = bool(os.environ.get("WANDB_API_KEY")) and device == "cuda"
    if use_wandb and cfg.logging.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", cfg.logging.wandb_project)
    args = SFTConfig(
        output_dir=str(Path(out_dir) / "checkpoints"),
        num_train_epochs=t.epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=t.lr,
        lr_scheduler_type=t.scheduler,
        warmup_steps=t.warmup_ratio,  # < 1 → fraction of total steps (transformers 5)
        per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum_steps,
        bf16=t.bf16 and device == "cuda",
        use_cpu=device == "cpu",
        seed=t.seed,
        data_seed=t.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        save_only_model=True,
        logging_steps=1,
        report_to=["wandb"] if use_wandb else "none",
        max_length=t.max_seq_len,
        assistant_only_loss=t.assistant_only_loss,
        packing=False,
        dataloader_pin_memory=device == "cuda",
        disable_tqdm=True,
    )
    lora = LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
    )  # fmt: skip

    def ds(part: str):
        return Dataset.from_list([{"messages": r["messages"]} for r in records[part]])

    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds("train"), eval_dataset=ds("dev"),
        processing_class=tokenizer, peft_config=lora,
    )  # fmt: skip
    trainer.gaitlab_settings = {  # type: ignore[attr-defined]
        "device": device, "dtype": settings.dtype,
        "attn_implementation": settings.attn_implementation, "model_id": model_id,
    }  # fmt: skip
    return trainer


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return out


def train(
    model_key: str,
    sft_dir: str = "data/sft",
    cfg_path: str = "configs/train_lora.yaml",
    out_root: str = "runs",
    allow_synthetic: bool = False,
    *,
    max_steps: int | None = None,
    device: str | None = None,
) -> Path:
    """See module docstring. Returns the run directory."""
    from gaitlab.config import config_hash, load_train_lora_config
    from gaitlab.llm import runtime
    from gaitlab.report.results import manifest_hash

    records = read_sft(sft_dir)
    synthetic = check_synthetic(records, allow_synthetic)
    cfg = load_train_lora_config(cfg_path)
    if model_key not in cfg.models:
        raise ValueError(f"model {model_key!r} not in {cfg_path} (have {sorted(cfg.models)})")
    resolved = cfg.model_dump(mode="json")
    chash = config_hash({**resolved, "max_steps": max_steps})
    run_id = make_run_id(model_key, dt.datetime.now(dt.UTC), chash)
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if synthetic:
        (run_dir / "SYNTHETIC").write_text(SYNTHETIC_MARKER)

    with runtime.offline_if_cached(cfg.models[model_key]):
        trainer = build_trainer(
            model_key, records, cfg, run_dir, device=device, max_steps=max_steps
        )
        trainer.train()
        trainer.save_model(str(run_dir / "adapter"))
    state = strip_timing(json.loads(json.dumps(dataclasses.asdict(trainer.state), default=str)))
    (run_dir / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
    (run_dir / "config.json").write_text(
        json.dumps({**resolved, "max_steps": max_steps}, indent=2) + "\n"
    )
    sft = Path(sft_dir)
    run_meta = {
        "run_id": run_id,
        "model_key": model_key,
        **trainer.gaitlab_settings,  # type: ignore[attr-defined]
        "config_hash": chash,
        "data_manifest_hash": manifest_hash([sft / "train.jsonl", sft / "dev.jsonl"]),
        "n_train": len(records["train"]),
        "n_dev": len(records["dev"]),
        "max_steps": max_steps,
        "global_step": trainer.state.global_step,
        "best_checkpoint_step": _step_of(trainer.state.best_model_checkpoint),
        "synthetic": synthetic,
        "versions": _versions(),
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    return run_dir


def _step_of(checkpoint: str | None) -> int | None:
    if not checkpoint:
        return None
    tail = Path(checkpoint).name.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def latest_run(runs_root: str | Path, model_key: str, synthetic: bool) -> Path:
    """Newest ``runs/<run_id>`` for ``model_key`` whose ``synthetic`` flag matches."""
    found = []
    for d in Path(runs_root).glob(f"{model_key}-*"):
        meta = d / "run.json"
        if meta.exists() and json.loads(meta.read_text()).get("synthetic") is synthetic:
            found.append(d)
    if not found:
        kind = "synthetic" if synthetic else "real"
        raise FileNotFoundError(f"no {kind} {model_key} run in {runs_root}; run gaitlab train")
    return max(found, key=lambda d: d.name)
