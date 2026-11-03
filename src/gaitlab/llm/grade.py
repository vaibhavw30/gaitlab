"""Grading with Gemma 3 + LoRA (TECHNICAL_PLAN §7.2, ADR-0002, ADR-0004). TASKS M3.

Contract:

``load_grader(model_id, adapter_path=None, dtype=torch.bfloat16, device="cuda") -> Grader``
    Load tokenizer + model (``AutoModelForCausalLM``), apply the PEFT adapter if given, and
    ``merge_and_unload()`` for inference. Set ``tokenizer.padding_side = "left"``.

``Grader.grade_token_ids() -> list[int]``
    Token ids for " 1".." 5" (from configs/train_lora.yaml ``grade.grade_tokens``). Assert each
    encodes to exactly one token with ``add_special_tokens=False``; if the leading-space form
    isn't single-token for this tokenizer, try the bare digit and record which in meta.

``Grader.scoring_inputs(prompts) -> dict[str, Tensor]``
    For each prompt: ``tokenizer.apply_chat_template([{"role":"user","content":p}],
    add_generation_prompt=True, tokenize=False) + GRADE_PREFIX``, then tokenize with
    ``padding="max_length", max_length=cfg.grade.max_prompt_tokens, truncation=False``.
    Raise if any prompt exceeds the max (never silently truncate features).

``Grader.score(prompts, batch_size) -> np.ndarray (N, 5)``
    One forward pass per batch; logits at the last position; restrict to the 5 grade ids;
    softmax in float32. This is the function benchmarked eager vs TensorRT (bench/trt.py) via
    ``ScoringWrapper`` below.

``Grader.generate_tags_and_cue(prompts, grades) -> list[ParsedOutput]``
    Continue from ``GRADE_PREFIX + f" {grade}\\n"`` with greedy decoding
    (``max_new_tokens`` from config) and parse with ``llm.prompt.parse_output``.

``ScoringWrapper(torch.nn.Module)``
    ``forward(input_ids, attention_mask) -> (B, 5) logits``. Wraps the merged model; no
    ``generate``, no KV cache, static shapes. This is what Torch-TensorRT compiles.

GPU-only tests are marked ``@pytest.mark.gpu``. A CPU unit test can cover the token-id assertion
using a small public tokenizer if one is available offline; otherwise mark it ``data``.
"""

from __future__ import annotations


def load_grader(model_id: str, adapter_path: str | None = None, device: str = "cuda"):
    """See module docstring."""
    raise NotImplementedError("TASKS.md M3: grader")
