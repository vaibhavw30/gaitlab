# ADR-0014: TensorRT bench details: static batches, worst-case parity, bf16 option

- Status: accepted; item 5's default (fp16) superseded by ADR-0015
- Date: 2026-11-04

## Context
ADR-0004 fixes what B2 measures: a Torch-TensorRT FP16 compile of the grade scoring wrapper against eager FP16 at batch 1 and 16, with a 100% argmax parity gate. Writing `bench/trt.py` raised details that ADR-0004 leaves open:
- The number of test prompts isn't a multiple of 16, and each compiled module has a static shape.
- Parity can differ between the batch-1 and batch-16 engines.
- Gemma 3 is commonly run in bf16 because its activations can overflow FP16. An overflow shows up as non-finite logits, and `argmax` of NaN still returns an index.

## Decision
1. **Static batches.** Prompts are cut into chunks of exactly the batch size, and the last chunk is padded by repeating its last prompt. Padding rows are dropped before parity is computed. Every test prompt is scored by both paths at every batch size.
2. **Worst-case parity.** Parity is computed per batch size against eager at the same precision. The reported `argmax_identical` is the minimum over batch sizes and `max_abs_dp` the maximum. Per-batch values are kept under `parity.by_batch_size`.
3. **Non-finite counts as failure.** A row with any non-finite probability counts as a disagreement, `n_nonfinite` is reported, and the gate fails if it is above zero.
4. **Gate outcome.** When the gate fails, the timings stay in the result, but the speedup fields are omitted and `speedup_reported` is false.
5. **Precision.** `configs/bench.yaml` `precision` accepts `fp16` (the default, as ADR-0004 says) or `bf16`. Changing it changes the config hash in the result. The README must name the precision used.
6. **Mismatch checks.** `gaitlab bench --trt` refuses when the `--synthetic` flag, the test prompts' `synthetic` flag and the adapter run's `synthetic` flag disagree.

## Consequences
- A failed FP16 parity is reported as a failure, not as a speedup, and it points at bf16 as the next config to try.
- No number in this ADR. B2 results come only from `results/bench/trt.json` (TBD).
