# ADR-0015: TensorRT bench precision is bf16, not FP16

- Status: accepted
- Date: 2026-11-07

## Context
ADR-0004 put the B2 bench in FP16, and ADR-0014 kept `fp16` as the `configs/bench.yaml` default, with bf16 as the next config to try if FP16 parity failed. Before spending more DeltaAI GPU time, the 270M scoring wrapper was run eagerly on CPU from the repo-local cache:
- In fp32 on an unpadded prompt, the residual stream (the decoder layer outputs) grows past the largest finite FP16 value from about the middle of the network onward. That's a property of the weights, not of the device or of padding.
- Cast to FP16, the grade logits are NaN. Cast to bf16, they are finite.
- `tests/test_bench.py::test_gemma3_overflows_fp16_so_the_bench_defaults_to_bf16` reproduces this whenever the 270M model is cached.

An FP16 bench would therefore always fail the parity gate on non-finite rows (ADR-0014 item 3), and the GPU run would be wasted. The 1B model wasn't checked on CPU (it isn't cached on the laptop); bf16 is safe for both.

## Decision
1. `configs/bench.yaml` and `BenchConfig` default `precision` to `bf16`. `fp16` stays selectable.
2. The eager reference uses the same precision as the TensorRT engine (unchanged from ADR-0014).
3. The README and the B2 result name the precision used (unchanged).

## Consequences
- The B2 speedup is TensorRT bf16 against eager bf16, not FP16 against FP16, and the README has to say so.
- The TensorRT runtime probe (`scripts/cluster/trt_probe.py`) gets bf16 cases, so the engine is checked at the precision the bench uses.
- The config hash of any bench result changes. There are no non-synthetic B2 results yet, so nothing has to be rerun.
- No number in this ADR. B2 results come only from `results/bench/trt.json` (TBD).
