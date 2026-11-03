"""Torch-TensorRT benchmark of the grade scoring pass (ADR-0004, EVALUATION B2). TASKS M4.

Contract:
``bench_scoring(grader, prompts, cfg_bench) -> dict``
1. Build ``ScoringWrapper`` (llm/grade.py) in FP16, eval mode, on CUDA.
2. Inputs: ``grader.scoring_inputs(prompts)`` padded to ``max_prompt_tokens``; batch sizes from
   ``configs/bench.yaml`` (1 and 16).
3. Compile: ``torch_tensorrt.compile(wrapper, ir="dynamo", arg_inputs=[ids, mask],
   use_explicit_typing=True, optimization_level=cfg.trt.optimization_level)``. Use explicit
   typing with an FP16 model (``enabled_precisions`` is deprecated in recent Torch-TensorRT).
   One compiled module per batch size (static shapes).
4. Time eager and compiled with ``bench.latency.cuda_time`` (warmup/iters from config).
5. Parity over all test prompts: argmax identical fraction, max |ΔP| after softmax. If parity <
   ``cfg.trt.parity_required``, set ``speedup_reported = False`` and omit speedup fields.
6. Return metrics for ``write_result("bench", "trt", ...)``: per batch size p50/p95/mean ms for
   both paths, speedups, parity, plus torch, torch_tensorrt and TensorRT versions (also captured
   by provenance).
"""

from __future__ import annotations


def bench_scoring(grader, prompts, cfg_bench) -> dict:
    raise NotImplementedError("TASKS.md M4: TensorRT benchmark")
