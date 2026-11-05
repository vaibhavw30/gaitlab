"""llm.grade and llm.runtime: grade tokens, device choice, logit scoring (TECHNICAL_PLAN §7.2).

The pure parts run everywhere. Tests that need the real Gemma 3 270M tokenizer or weights skip
unless they are in the repo-local Hugging Face cache (``.hf_cache/``, CLAUDE.md rule 8).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from gaitlab.llm import runtime
from gaitlab.llm.grade import (
    GradeTokenError,
    check_prefix_consistency,
    choose_grade_tokens,
    llm_model_name,
)
from gaitlab.llm.prompt import GRADE_PREFIX

GRADE_TOKENS = [" 1", " 2", " 3", " 4", " 5"]
MODEL_270M = "google/gemma-3-270m-it"

has_llm = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))
requires_270m = pytest.mark.skipif(
    not has_llm or not runtime.is_cached(MODEL_270M),
    reason="needs the 'llm' extra and gemma-3-270m-it in .hf_cache/ (see README)",
)


class _Vocab:
    """A toy encoder: known strings map to one id, anything else splits per character."""

    def __init__(self, whole: list[str]):
        self.ids = {s: i for i, s in enumerate(whole, start=100)}

    def __call__(self, text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for s in sorted(self.ids, key=len, reverse=True):
                if text.startswith(s, i):
                    out.append(self.ids[s])
                    i += len(s)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out


def test_leading_space_digits_are_used_when_single_tokens():
    enc = _Vocab(["grade:", *GRADE_TOKENS])
    gt = choose_grade_tokens(enc, GRADE_TOKENS)
    assert gt.form == "leading_space"
    assert gt.strings == tuple(GRADE_TOKENS)
    assert gt.score_prefix == GRADE_PREFIX
    assert gt.ids == tuple(enc(s)[0] for s in GRADE_TOKENS)


def test_bare_digits_are_the_fallback_when_space_digit_splits():
    # Gemma 3 style: " 3" encodes as [" ", "3"], so the space moves into the scoring prefix.
    enc = _Vocab(["grade:", "1", "2", "3", "4", "5"])
    gt = choose_grade_tokens(enc, GRADE_TOKENS)
    assert gt.form == "bare_digit"
    assert gt.strings == ("1", "2", "3", "4", "5")
    assert gt.score_prefix == GRADE_PREFIX + " "
    check_prefix_consistency(enc, "<turn>model\n", gt)


def test_no_single_token_form_raises():
    def split_everything(text: str) -> list[int]:
        return [ord(c) for c in text] + [0]  # every string is ≥ 2 tokens

    with pytest.raises(GradeTokenError):
        choose_grade_tokens(split_everything, GRADE_TOKENS)


def test_prefix_consistency_catches_a_merge_across_the_boundary():
    # " 3" alone is one token, but after "grade" the tokenizer merges ": 3", so the scored
    # position would not be where training put the digit.
    enc = _Vocab(["grade", ": 3", *GRADE_TOKENS])
    gt = choose_grade_tokens(enc, GRADE_TOKENS)
    assert gt.form == "leading_space"
    with pytest.raises(GradeTokenError, match="3"):
        check_prefix_consistency(enc, "", gt)


@pytest.mark.parametrize(
    ("cuda", "mps", "want"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu"), (True, False, "cuda")],
)
def test_device_priority_is_cuda_then_mps_then_cpu(cuda: bool, mps: bool, want: str):
    assert runtime.select_device(cuda_available=cuda, mps_available=mps) == want


def test_explicit_device_wins_and_is_validated():
    assert runtime.select_device(cuda_available=True, mps_available=True, prefer="cpu") == "cpu"
    with pytest.raises(ValueError, match="cuda"):
        runtime.select_device(cuda_available=False, mps_available=True, prefer="cuda")
    with pytest.raises(ValueError, match="tpu"):
        runtime.select_device(cuda_available=False, mps_available=False, prefer="tpu")


def test_mps_and_cpu_load_float32_eager():
    for dev in ("mps", "cpu"):
        s = runtime.load_settings(dev, training=True)
        assert (s.dtype, s.attn_implementation) == ("float32", "eager")
        assert runtime.load_settings(dev, training=False).dtype == "float32"
    assert runtime.load_settings("cuda", training=True) == runtime.LoadSettings("bfloat16", "eager")


def test_hf_cache_stays_inside_the_repo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    root = runtime.repo_root()
    assert runtime.hf_home().is_relative_to(root)
    assert runtime.hf_cache_dir() == runtime.hf_home() / "hub"


def test_model_names_match_the_ladder():
    from gaitlab.eval.run_eval import LADDER_ORDER

    names = {llm_model_name(k, zero_shot=z) for k in ("1b", "270m") for z in (True, False)}
    assert {"gemma3_1b_zeroshot", "gemma3_1b_lora", "gemma3_270m_lora"} <= names
    assert names - {"gemma3_270m_zeroshot"} <= set(LADDER_ORDER)


def test_llm_modules_import_without_torch():
    code = (
        "import sys, gaitlab.llm.grade, gaitlab.llm.train_lora, gaitlab.llm.runtime; "
        "bad = {'torch', 'transformers', 'peft', 'trl', 'datasets'} & set(sys.modules); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# --- real Gemma 3 270M tokenizer / weights --------------------------------------------------


@pytest.mark.own_process
@requires_270m
def test_gemma3_splits_space_digit_so_bare_digits_are_scored():
    from gaitlab.llm.grade import grade_tokens_for, load_tokenizer

    tok = load_tokenizer(MODEL_270M)
    gt = grade_tokens_for(tok, GRADE_TOKENS)
    assert gt.form == "bare_digit"
    for s, i in zip(gt.strings, gt.ids, strict=True):
        assert tok.encode(s, add_special_tokens=False) == [i]


@pytest.mark.slow
@pytest.mark.own_process
@requires_270m
def test_score_and_generate_on_cpu(synthetic_features_cfg):
    from gaitlab.config import load_train_lora_config
    from gaitlab.llm.grade import load_grader
    from gaitlab.llm.prompt import format_prompt

    cfg = load_train_lora_config("configs/train_lora.yaml")
    g = load_grader(MODEL_270M, cfg=cfg, device="cpu")
    assert g.tokenizer.padding_side == "left"
    assert g.meta()["grade_token_form"] == "bare_digit"
    row = {"speed_kmh": 12.0, "cadence_spm_med": 165.0, "overstride_L_med": 0.3}
    prompts = [format_prompt(row, synthetic_features_cfg.feature_order)] * 2
    inputs = g.scoring_inputs(prompts)
    assert tuple(inputs["input_ids"].shape) == (2, cfg.grade.max_prompt_tokens)
    assert (inputs["attention_mask"][:, -1] == 1).all(), "left padding: last position is real"
    p = g.score(prompts, batch_size=2)
    assert p.shape == (2, 5)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(p[0], p[1], atol=1e-5)
    out = g.generate_tags_and_cue(prompts[:1], [3])
    assert len(out) == 1 and out[0].grade == 3 and isinstance(out[0].cue, str)

    too_long = format_prompt(row, synthetic_features_cfg.feature_order) + " x" * 400
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        g.scoring_inputs([too_long])


@pytest.mark.slow
@pytest.mark.own_process
@requires_270m
def test_scoring_wrapper_matches_score():
    import torch

    from gaitlab.config import load_train_lora_config
    from gaitlab.llm.grade import load_grader

    g = load_grader(MODEL_270M, cfg=load_train_lora_config(), device="cpu")
    prompts = ["Grade this treadmill running window from 1 (poor) to 5 (excellent)."]
    wrapper = g.scoring_wrapper()
    inp = g.scoring_inputs(prompts)
    with torch.inference_mode():
        logits = wrapper(inp["input_ids"], inp["attention_mask"])
    assert tuple(logits.shape) == (1, 5)
    np.testing.assert_allclose(
        torch.softmax(logits.float(), -1).numpy(), g.score(prompts), atol=1e-5
    )


@pytest.mark.gpu
@pytest.mark.own_process
@requires_270m
def test_cuda_scoring_matches_cpu_argmax():
    from gaitlab.config import load_train_lora_config
    from gaitlab.llm.grade import load_grader

    prompts = [f"cadence_spm: {150 + 5 * i}.0 (iqr 2.0)" for i in range(8)]
    cfg = load_train_lora_config()
    gpu = load_grader(MODEL_270M, cfg=cfg, device="cuda")
    assert gpu.dtype == "bfloat16"
    cpu = load_grader(MODEL_270M, cfg=cfg, device="cpu")
    np.testing.assert_array_equal(gpu.score(prompts).argmax(1), cpu.score(prompts).argmax(1))
