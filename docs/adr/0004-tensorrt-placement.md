# ADR-0004: Where TensorRT goes

- Status: accepted
- Date: 2026-11-03

## Context
Torch-TensorRT's Hugging Face LLM tooling lists GPT-2, Llama 3.x and Qwen 2.5 families, not Gemma. TensorRT-LLM deprecated Windows at v0.18. The grade comes from one forward pass (ADR-0002).

## Decision
1. Compile a wrapper module that returns the 5 grade logits for a left-padded, fixed-length batch with `torch_tensorrt.compile(ir="dynamo")` in FP16. Benchmark it against eager FP16 at batch 1 and 16.
2. (Stretch) Build a TensorRT FP16 engine for the RTMPose ONNX model and benchmark it against ONNX Runtime CUDA.
3. Parity gate: argmax grade identical on 100% of test prompts, or no speedup is reported.

## Consequences
Cue generation stays in eager PyTorch. The reported speedup is specifically "grade scoring pass". The README must say exactly that.
