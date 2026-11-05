# ADR-0013: LLM grader: Gemma 3 grade tokens, device choice, local model cache, Mac dry runs

- Status: accepted
- Date: 2026-11-04

## Context
ADR-0002 scores the grade from the logits of " 1"…" 5" after `grade:` and asks for a unit test that each is a single token. The Gemma 3 tokenizer (checked on `google/gemma-3-270m-it`) encodes " 3" as two tokens, " " + "3"; the bare digits are single tokens, and `grade: 3` encodes as `grade`, `:`, ` `, `3`.

The GPU host for M3 is still undecided (TASKS M3), so `llm/grade.py` and `llm/train_lora.py` have to run on a CPU-only Mac for dry runs as well as on CUDA. Model weights are a gated Hugging Face download that defaults to `~/.cache/huggingface`, outside the repo (CLAUDE.md rule 8). On macOS, LightGBM and torch each bring their own OpenMP runtime, and loading both into one process segfaults (seen in either import order).

## Decision
1. **Grade tokens:** use " 1"…" 5" if each is one token. Otherwise use the bare digits "1"…"5" and move the space into the scoring prefix (`"grade: "`). Either way, assert on the real chat template that `encode(context + prefix + digit) == encode(context + prefix) + [digit id]`, so the scored position is where training put the digit. Both checks run on the tokenizer before any weights load. The form used is recorded in the predictions' `meta.json` (`grade_token_form`). Gemma 3 uses `bare_digit`. The SFT targets are unchanged (`grade: 3`), so training needs nothing new.
2. **Device:** `cuda` > `mps` > `cpu`, overridable with `--device`. CUDA loads bfloat16; MPS and CPU load float32. Training uses eager attention everywhere (the Gemma 3 recommendation); inference uses eager on MPS/CPU and SDPA on CUDA.
3. **Assistant-only loss:** Gemma 3's chat template has no `{% generation %}` markers. TRL's `assistant_only_loss=True` swaps in its Gemma 3 training template; a unit test on one batch checks that the user prompt carries no loss and that the assistant turn, including `<end_of_turn>`, does.
4. **Model cache in the repo:** the Hugging Face cache defaults to `<repo>/.hf_cache` (git-ignored) unless `HF_HOME` is set, and every `from_pretrained` call gets that `cache_dir` and `local_files_only` once the files are present. SFT data goes through in-memory `datasets.Dataset`s, so nothing is cached outside the repo. Downloads need `HF_TOKEN` in the environment.
5. **G6 sampling:** tags and cue are sampled at the configured temperature with no top-k or top-p, seeded `seed .. seed+runs-1`. The grade comes from repeated scoring passes. Repeats are written to `<model>.consistency.json` next to the predictions. The verifier (G4) runs in `eval`, from the window features and the configured thresholds.
6. **No timing from the Mac:** training drops `*_runtime` and `*_per_second` from the saved trainer state, `predict` records none, and W&B logging only turns on for CUDA runs. Speed numbers come only from `bench/` on the GPU host.
7. **Process separation:** LightGBM (`baselines`) and torch (`train`, `predict`) never share a process. They are separate CLI commands, and tests that chain them run the torch steps in subprocesses.

## Consequences
- Logit grading works unchanged for Gemma 3, and a tokenizer where neither form works fails loudly before any weights load.
- Synthetic LoRA runs on a Mac are plumbing checks only. A float32 run on MPS is not numerically the bf16 CUDA run, and nothing from it is reported.
- The TensorRT scoring wrapper (ADR-0004) is unaffected: it still returns five logits from one static-shape forward pass.
- A real `predict` and `eval` (without `--synthetic`) still wait on real labels and the frozen split (M2).
