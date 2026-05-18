# Experiment 04 — Geoclassifier: ONNX vs TensorRT FP16

## Setup
- `geoclassifier`: EfficientNet-V2-M fine-tuned for 17 Quebec administrative regions
- Two backends tested: ONNX FP32 and TensorRT FP16 (converted with `trtexec`)
- Pipeline: C++ preprocess (resize 512 → center-crop 480 → ImageNet normalize) → model → Python postprocess (softmax → argmax → label + confidence, no NMS)

## Notes on TRT conversion
- Model was exported with PyTorch 2.10, which produces opset 18 and IR version 10
- Required two fixes before trtexec would accept the model:
  1. IR version downgraded to 8 (`model.ir_version = 8`, saved with `save_as_external_data=False`)
  2. `onnxsim` to resolve dynamic shapes in the SE block
  3. Re-exported with `opset_version=12` and `dynamo=False` — newer exporters produce a graph structure TRT 8.6 cannot parse for EfficientNet-V2's SqueezeExcitation blocks

## How tests were run

**Model-only** (perf_analyzer, inside Triton SDK container):
```bash
bash benchmarks/04_geoclassifier/benchmark.sh geoclassifier
bash benchmarks/04_geoclassifier/benchmark.sh geoclassifier_trt
```

**Pipeline** (load_test.py, from repo root on host):
```bash
bash benchmarks/04_geoclassifier/run_pipeline.sh
```

Concurrency sweep: 1–8 | Duration: 30s per level | Protocol: HTTP

## Results

### Model-only (GPU compute isolated)

| Model | p50 @c=1 | p99 @c=1 | GPU infer time | Peak throughput | Saturates at |
|-------|----------|----------|----------------|-----------------|--------------|
| geoclassifier (ONNX FP32) | 42ms | 43ms | 37ms | 26 inf/s | c=2 |
| geoclassifier_trt (TRT FP16) | 46ms | 49ms | 28ms | 35 inf/s | c=2 |

TRT GPU infer time is **25% faster** (28ms vs 37ms). But total latency at c=1 is **9% slower** (46ms vs 42ms) — TRT has higher per-request overhead in non-compute operations (network + server I/O: ~17ms vs ~5ms). This overhead is fixed per request; it becomes negligible as the GPU is kept busy.

TRT peak throughput is **35% higher** (35 vs 26 inf/s).

### Pipeline (C++ preprocess + model + Python postprocess)

| Pipeline | p50 @c=1 | p99 @c=1 | Peak throughput | Saturates at |
|----------|----------|----------|-----------------|--------------|
| geoclassifier_pipeline (ONNX) | 46ms | 47ms | 26 inf/s | c=2 |
| geoclassifier_trt_pipeline (TRT FP16) | 58ms | 62ms | 31 inf/s | c=3 |

TRT pipeline peak is **21% higher** (31 vs 26 inf/s). TRT's GPU advantage propagates through to end-to-end throughput — both pipelines scale differently and TRT saturates later at a higher ceiling.

The pre/postprocess adds almost no overhead: ONNX pipeline at c=1 is 46ms vs 42ms model-only — only 4ms for preprocess + postprocess combined.

## Key takeaways

- **TRT is faster under load, slower per single request.** The per-request overhead (CUDA context, memory management) is visible at c=1 but amortises at higher concurrency. For latency-sensitive single-request serving of this model, ONNX Runtime is the better choice.
- **Pascal (GTX 1060, compute 6.1) limits FP16 gains.** The 25% GPU compute improvement is real but modest compared to what FP16 yields on Volta/Turing/Ampere. The throughput advantage at peak (35%) is meaningful but not dramatic.
- **TRT compatibility with newer exporters is fragile.** EfficientNet-V2-M required three separate fixes to parse. YOLOv8s converted cleanly — architectures with SE blocks and adaptive pooling hit edge cases in TRT 8.6's ONNX importer.
- **TRT's GPU advantage is visible end-to-end.** Both pipelines scale differently — TRT saturates later at a higher ceiling (31 vs 26 inf/s).
- **Both models fit on 3GB VRAM simultaneously** — sets up experiment 05 (multi-model deployment).

## Hardware
- GPU: GTX 1060 3GB
- CPU: 4 cores / 8 threads
- Triton: 23.08, TensorRT 8.6
