# ADR-0002: Read the grade from next-token probabilities

- Status: accepted
- Date: 2026-11-03

## Context
Parsing a generated grade can fail, gives no probability, and costs a full decode.

## Decision
Prefill the assistant turn with `grade:`, run one forward pass, take the logits for the tokens " 1"…" 5" at the last position, and softmax them. A unit test asserts each is a single token for the tokenizer in use. Tags and the cue are generated after that, greedily.

## Consequences
- Zero parse failures; calibration metrics (ECE, Brier) become possible.
- The scoring pass is a fixed-shape forward pass, which suits TensorRT (ADR-0004).
- If a tokenizer splits a digit token, this ADR must be revisited. Gemma 3 splits " 1" into " " + "1": see [ADR-0013](0013-llm-grader-on-cpu-mps-and-gemma3-grade-tokens.md) for the bare-digit fallback.
