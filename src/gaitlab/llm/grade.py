"""Grading with Gemma 3 + LoRA (TECHNICAL_PLAN §7.2, ADR-0002, ADR-0004, ADR-0013). TASKS M3.

Contract:

``load_grader(model_id, adapter_path=None, *, cfg, device=None, dtype=None) -> Grader``
    Load the tokenizer first and settle the grade tokens (below) before loading any weights.
    Then the model (``AutoModelForCausalLM``; device cuda > mps > cpu, bfloat16 on CUDA and
    float32 with eager attention on MPS/CPU, ``llm.runtime``), apply the PEFT adapter if given,
    and ``merge_and_unload()`` for inference. ``tokenizer.padding_side = "left"``.

``Grader.grade_token_ids() -> list[int]``
    Token ids for the five grades (configs/train_lora.yaml ``grade.grade_tokens``, " 1".." 5").
    Each must encode to exactly one token with ``add_special_tokens=False``. If the leading-space
    form isn't single-token for this tokenizer, the bare digits are tried and the space moves into
    the scoring prefix (``"grade: "``). The form used is recorded in ``Grader.meta()``. Either way
    ``encode(context + prefix + token) == encode(context + prefix) + [id]`` is asserted on the
    real chat template, so the scored position is where training put the digit. Gemma 3 splits
    " 3" into " " + "3", so it uses the bare-digit form (ADR-0013).

``Grader.scoring_inputs(prompts) -> dict[str, Tensor]``
    For each prompt: ``tokenizer.apply_chat_template([{"role":"user","content":p}],
    add_generation_prompt=True, tokenize=False) + score_prefix``, then tokenize with
    ``padding="max_length", max_length=cfg.grade.max_prompt_tokens, truncation=False``.
    Raises if any prompt exceeds the max (never silently truncate features).

``Grader.score(prompts, batch_size) -> np.ndarray (N, 5)``
    One forward pass per batch; logits at the last position; restricted to the 5 grade ids;
    softmax in float32. This is the function benchmarked eager vs TensorRT (bench/trt.py) via
    ``Grader.scoring_wrapper()``.

``Grader.generate_tags_and_cue(prompts, grades, *, do_sample=False, temperature=None, seed=None)``
    Continue from ``GRADE_PREFIX + f" {grade}\\n"`` with greedy decoding by default
    (``max_new_tokens`` from config) and parse with ``llm.prompt.parse_output``. Sampling (for
    G6 consistency) uses plain temperature sampling, no top-k/top-p, seeded per call.

``Grader.scoring_wrapper() -> torch.nn.Module``
    ``forward(input_ids, attention_mask) -> (B, 5) logits``. Wraps the merged model; no
    ``generate``, no KV cache, static shapes. This is what Torch-TensorRT compiles.

``predict_windows(grader, windows, feature_order, ...) -> (rows, consistency)``
    Prediction rows in the ``eval.run_eval.PRED_COLUMNS`` schema plus the per-window repeats
    for G6. File writing stays in ``cli.py``.

torch, transformers and peft are imported inside functions (``llm`` extra).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gaitlab.llm.prompt import GRADE_PREFIX, ParsedOutput, format_prompt, parse_output

Encode = Callable[[str], list[int]]


class GradeTokenError(AssertionError):
    """The grade digits can't be scored as single next tokens with this tokenizer."""


@dataclass(frozen=True)
class GradeTokens:
    form: str  # "leading_space" | "bare_digit"
    strings: tuple[str, ...]  # the token strings scored, grades 1..5
    ids: tuple[int, ...]
    score_prefix: str  # appended after the rendered chat: "grade:" or "grade: "


def choose_grade_tokens(encode: Encode, grade_tokens: Sequence[str]) -> GradeTokens:
    """Leading-space digits if each is one token, else bare digits; raise if neither works."""
    for form, strings, prefix in (
        ("leading_space", tuple(grade_tokens), GRADE_PREFIX),
        ("bare_digit", tuple(s.strip() for s in grade_tokens), GRADE_PREFIX + " "),
    ):
        enc = [encode(s) for s in strings]
        if all(len(e) == 1 for e in enc):
            ids = tuple(e[0] for e in enc)
            if len(set(ids)) != len(ids):
                raise GradeTokenError(f"grade tokens {strings} share ids {ids}")
            return GradeTokens(form, strings, ids, prefix)
    raise GradeTokenError(
        f"neither {list(grade_tokens)} nor the bare digits encode to single tokens: "
        f"{[encode(s) for s in grade_tokens]}"
    )


def check_prefix_consistency(encode: Encode, context: str, gt: GradeTokens) -> None:
    """Assert the grade token follows the scoring prefix unchanged inside real context."""
    base = encode(context + gt.score_prefix)
    for s, i in zip(gt.strings, gt.ids, strict=True):
        full = encode(context + gt.score_prefix + s)
        if full != [*base, i]:
            raise GradeTokenError(
                f"grade token {s!r} (id {i}) does not follow {gt.score_prefix!r} as one token: "
                f"tail {full[len(base) - 1 :]}"
            )


def llm_model_name(model_key: str, zero_shot: bool) -> str:
    """Prediction-file / ladder name, e.g. ``gemma3_270m_lora``."""
    return f"gemma3_{model_key}_{'zeroshot' if zero_shot else 'lora'}"


def load_tokenizer(model_id: str):
    from gaitlab.llm import runtime

    runtime.use_repo_hf_home()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, **runtime.from_pretrained_kwargs(model_id))


def _encoder(tokenizer) -> Encode:
    return lambda text: list(tokenizer.encode(text, add_special_tokens=False))


def render_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )


def grade_tokens_for(tokenizer, grade_tokens: Sequence[str]) -> GradeTokens:
    """Choose the grade-token form and check it inside this tokenizer's chat template."""
    enc = _encoder(tokenizer)
    gt = choose_grade_tokens(enc, grade_tokens)
    check_prefix_consistency(enc, render_chat(tokenizer, "cadence_spm: 170.0 (iqr 2.0)"), gt)
    return gt


@dataclass
class Grader:
    model: Any
    tokenizer: Any
    cfg: Any  # config.TrainLoraConfig
    tokens: GradeTokens
    device: str
    dtype: str
    attn_implementation: str
    model_id: str
    adapter_path: str | None = None

    def meta(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "adapter_path": self.adapter_path,
            "grade_token_form": self.tokens.form,
            "grade_token_strings": list(self.tokens.strings),
            "grade_token_ids": list(self.tokens.ids),
            "score_prefix": self.tokens.score_prefix,
            "device": self.device,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "max_prompt_tokens": self.cfg.grade.max_prompt_tokens,
        }

    def grade_token_ids(self) -> list[int]:
        return list(self.tokens.ids)

    def scoring_inputs(self, prompts: Sequence[str]) -> dict[str, Any]:
        max_len = self.cfg.grade.max_prompt_tokens
        texts = [render_chat(self.tokenizer, p) + self.tokens.score_prefix for p in prompts]
        lengths = [len(self.tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        if (n := max(lengths)) > max_len:
            raise ValueError(
                f"prompt is {n} tokens > grade.max_prompt_tokens={max_len}; raise the limit in "
                "configs/train_lora.yaml rather than truncating features"
            )
        return dict(
            self.tokenizer(
                texts, padding="max_length", max_length=max_len, truncation=False,
                add_special_tokens=False, return_tensors="pt",
            )
        )  # fmt: skip

    def scoring_wrapper(self):
        return _make_scoring_wrapper(self.model, self.tokens.ids)

    def score(self, prompts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        import torch

        wrapper = self.scoring_wrapper()
        out = []
        with torch.inference_mode():
            for i in range(0, len(prompts), batch_size):
                enc = self.scoring_inputs(prompts[i : i + batch_size])
                logits = wrapper(
                    enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
                )
                out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 5))

    def generate_tags_and_cue(
        self,
        prompts: Sequence[str],
        grades: Sequence[int],
        *,
        batch_size: int = 16,
        do_sample: bool = False,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> list[ParsedOutput]:
        import torch
        from transformers import GenerationConfig

        gen_cfg = self.model.generation_config
        g = GenerationConfig(
            max_new_tokens=self.cfg.generate.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=0 if do_sample else None,
            top_p=1.0 if do_sample else None,
            eos_token_id=gen_cfg.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=gen_cfg.bos_token_id,
        )
        if seed is not None:
            torch.manual_seed(seed)
        out: list[ParsedOutput] = []
        for i in range(0, len(prompts), batch_size):
            heads = [f"{GRADE_PREFIX} {int(gr)}\n" for gr in grades[i : i + batch_size]]
            texts = [
                render_chat(self.tokenizer, p) + h
                for p, h in zip(prompts[i : i + batch_size], heads, strict=True)
            ]
            enc = self.tokenizer(
                texts, padding=True, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                ids = self.model.generate(**enc, generation_config=g)
            new = ids[:, enc["input_ids"].shape[1] :]
            for h, text in zip(
                heads, self.tokenizer.batch_decode(new, skip_special_tokens=True), strict=True
            ):
                out.append(parse_output(h + text))
        return out


def _make_scoring_wrapper(model, grade_ids: Sequence[int]):
    import torch

    class ScoringWrapper(torch.nn.Module):
        """(input_ids, attention_mask) → (B, 5) grade logits at the last position."""

        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("grade_ids", torch.tensor(list(grade_ids)), persistent=False)

        def forward(self, input_ids, attention_mask):
            # Left padding: positions count real tokens only, as in training.
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                use_cache=False, logits_to_keep=1,
            )  # fmt: skip
            return out.logits[:, -1, :].index_select(-1, self.grade_ids)

    return ScoringWrapper().to(next(model.parameters()).device).eval()


def load_grader(
    model_id: str,
    adapter_path: str | None = None,
    *,
    cfg=None,
    device: str | None = None,
    dtype: str | None = None,
) -> Grader:
    """See module docstring. Grade tokens are checked before the weights are loaded."""
    from gaitlab.config import load_train_lora_config
    from gaitlab.llm import runtime

    cfg = cfg or load_train_lora_config()
    tokenizer = load_tokenizer(model_id)
    tokens = grade_tokens_for(tokenizer, cfg.grade.grade_tokens)
    tokenizer.padding_side = "left"

    from transformers import AutoModelForCausalLM

    device = runtime.pick_device(device)
    settings = runtime.load_settings(device, training=False)
    dtype = dtype or settings.dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=runtime.torch_dtype(dtype),
        attn_implementation=settings.attn_implementation,
        **runtime.from_pretrained_kwargs(model_id),
    )  # fmt: skip
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.to(device).eval()
    return Grader(
        model, tokenizer, cfg, tokens, device, dtype, settings.attn_implementation, model_id,
        adapter_path,
    )  # fmt: skip


def _pred_row(wid: str, probs: np.ndarray, parsed: ParsedOutput, synthetic: bool) -> dict:
    row: dict[str, Any] = {"window_id": wid, "pred_grade": int(np.argmax(probs)) + 1}
    for k in range(5):
        row[f"p{k + 1}"] = float(probs[k])
    row["tags"] = ";".join(parsed.tags)
    row["cue"] = parsed.cue
    row["synthetic"] = synthetic
    return row


def predict_windows(
    grader: Grader,
    windows: Sequence[Mapping[str, Any]],
    feature_order: Sequence[str],
    *,
    synthetic: bool,
    consistency_runs: int = 0,
    temperature: float = 0.7,
    seed: int = 0,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Logit grades + greedy tags/cue per window; optional G6 repeats.

    With ``consistency_runs = R > 0`` the scoring pass is repeated R times (greedy grade per
    run) and tags + cue are sampled R times at ``temperature`` with seeds ``seed .. seed+R-1``.
    """
    ids = [str(w["window_id"]) for w in windows]
    prompts = [format_prompt(w, feature_order) for w in windows]
    probs = grader.score(prompts, batch_size)
    grades = [int(g) for g in probs.argmax(axis=1) + 1]
    parsed = grader.generate_tags_and_cue(prompts, grades, batch_size=batch_size)
    rows = [_pred_row(w, p, o, synthetic) for w, p, o in zip(ids, probs, parsed, strict=True)]
    if consistency_runs <= 0:
        return rows, None
    reps: dict[str, dict[str, list]] = {w: {"grades": [], "tags": [], "cues": []} for w in ids}
    for r in range(consistency_runs):
        g_r = grader.score(prompts, batch_size).argmax(axis=1) + 1
        s_r = grader.generate_tags_and_cue(
            prompts, [int(x) for x in g_r], batch_size=batch_size, do_sample=True,
            temperature=temperature, seed=seed + r,
        )  # fmt: skip
        for w, g, o in zip(ids, g_r, s_r, strict=True):
            reps[w]["grades"].append(int(g))
            reps[w]["tags"].append(sorted(o.tags))
            reps[w]["cues"].append(o.cue)
    seeds = list(range(seed, seed + consistency_runs))
    meta = {"runs": consistency_runs, "temperature": temperature, "seeds": seeds}
    return rows, {**meta, "synthetic": synthetic, "windows": reps}
