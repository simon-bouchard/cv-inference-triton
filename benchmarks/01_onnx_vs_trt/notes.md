# Experiment 01 — ONNX vs TensorRT

## What changed
- Converted `yolov8s` ONNX model to TensorRT FP16 using `trtexec`
- Created `yolov8s_trt` as a separate model in the repository alongside the original
- Created `yolov8s_trt_pipeline` ensemble routing through the TRT model
- Increased Python instance count to 2 for both `preprocess` and `postprocess` models

## How tests were run

**Model-only** (perf_analyzer, inside Triton SDK container):
```bash
bash benchmarks/01_onnx_vs_trt/benchmark.sh yolov8s model
bash benchmarks/01_onnx_vs_trt/benchmark.sh yolov8s_trt model
```

**Pipeline** (load_test.py, from repo root on host):
```bash
bash benchmarks/01_onnx_vs_trt/run_pipeline.sh
```

Concurrency sweep: 1–8 | Duration: 30s per level | Protocol: HTTP

## Results

### Model-only (GPU compute isolated)

| Model | p50 latency | p99 latency | Throughput | Saturates at |
|-------|-------------|-------------|------------|--------------|
| yolov8s (ONNX) | 26ms | 29ms | 58 inf/s | c=2 |
| yolov8s_trt (TRT FP16) | 22ms | 28ms | 88 inf/s | c=3 |

TRT FP16 gives a **33% reduction in GPU compute time** and **~50% throughput gain** in isolation.

### Pipeline — 1 Python instance (baseline)

| Model | p50 @c=1 | p99 @c=1 | Peak throughput | Saturates at |
|-------|----------|----------|-----------------|--------------|
| yolov8s_pipeline | 78ms | 85ms | 17 inf/s | c=2 |
| yolov8s_trt_pipeline | 73ms | 78ms | 17 inf/s | c=2 |

Pipeline throughput is **3x lower** than model-only. The TRT advantage disappears entirely — both pipelines cap at ~17 inf/s. Bottleneck is the Python GIL serializing pre/postprocess workers.

### Pipeline — 2 Python instances

| Model | p50 @c=1 | p99 @c=1 | Peak throughput | Saturates at |
|-------|----------|----------|-----------------|--------------|
| yolov8s_pipeline | 78ms | 90ms | 33 inf/s | c=3 |
| yolov8s_trt_pipeline | 75ms | 79ms | 33 inf/s | c=3 |

> Note: both pipelines cap at the same throughput — TRT's GPU advantage is hidden by the Python pre/postprocess bottleneck.

Doubling Python instances doubled throughput, confirming GIL as the bottleneck. TRT advantage still hidden — pre/postprocess is now the ceiling.

## Key takeaways

- TRT FP16 gives a meaningful GPU speedup but end-to-end gains depend on the pipeline
- Python pre/postprocess is the current bottleneck, not the GPU
- The GPU is underutilised at ~33 inf/s — capacity exists for additional models
- Next bottleneck to address: Python overhead (C++ backend) or GPU utilisation (dynamic batching for multi-model scenarios)
- gRPC vs HTTP shows no measurable difference at current throughput levels — protocol overhead is negligible compared to Python pre/postprocess latency. Worth retesting after C++ backend implementation.

## Hardware
- GPU: GTX 1060 3GB
- CPU: 4 cores / 8 threads
- Triton: 23.08, TensorRT 8.6
