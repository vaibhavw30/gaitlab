"""LoRA fine-tuning (TECHNICAL_PLAN §7.2, configs/train_lora.yaml). TASKS M3.

Contract:
``train(model_key: Literal["1b", "270m"], sft_dir="data/sft", cfg_path="configs/train_lora.yaml",
        out_root="runs") -> Path``
- Load train/dev JSONL (``datasets.load_dataset("json", ...)``). Refuse to run if any record has
  ``synthetic=True`` unless ``allow_synthetic=True`` (dry runs); propagate the flag to the run dir.
- ``peft.LoraConfig`` from the YAML; ``trl.SFTTrainer`` with the chat-format ``messages`` field and
  assistant-only loss (use the TRL option for completion/assistant-only loss available in the
  installed version; verify that prompt tokens are masked with a unit test on one batch).
- Seeds fixed; bf16; eval on dev each epoch; keep the best by dev loss.
- Write ``runs/<run_id>/`` with: adapter weights, resolved config, ``config_hash``, train/dev
  manifest hash, ``trainer_state.json``, and ``synthetic`` flag. ``run_id`` =
  ``<model_key>-<UTC timestamp>-<short config hash>``.
- Optional W&B logging if ``WANDB_API_KEY`` is set; never required.
GPU smoke test (``@pytest.mark.gpu``): 4 optimizer steps on 8 synthetic records with 270m.
"""

from __future__ import annotations

from pathlib import Path


def train(model_key: str, sft_dir: str = "data/sft", cfg_path: str = "configs/train_lora.yaml",
          out_root: str = "runs", allow_synthetic: bool = False) -> Path:
    raise NotImplementedError("TASKS.md M3: LoRA training")
