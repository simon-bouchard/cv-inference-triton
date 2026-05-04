# Experiment 02 — C++ Preprocess Backend

## What changed
- Added `preprocess_cpp` Triton backend: C++ shared library that decodes JPEG and runs bilinear resize + normalize + HWC→CHW in a single pass, with no Python interpreter overhead
- Added `yolov8s_trt_pipeline_cpp` ensemble routing through `preprocess_cpp` → `yolov8s_trt` → `postprocess`

## Hypothesis
Python GIL was the bottleneck in experiment 01 (pipeline capped at ~33 inf/s with 2 instances while GPU was underutilised). Moving preprocess to C++ should remove the GIL entirely and let the pipeline saturate the GPU instead.

## How tests were run

**Preprocess in isolation** (perf_analyzer, inside Triton SDK container):
```bash
bash benchmarks/02_cpp_backend/benchmark.sh preprocess
```

**Full pipeline** (load_test.py, from repo root on host):
```bash
bash benchmarks/02_cpp_backend/run_pipeline.sh
```

Concurrency sweep: 1–8 | Duration: 30s per level | Protocol: HTTP

## Results

### Preprocess in isolation

| Backend | p50 latency | p99 latency | Throughput | Saturates at |
|---------|-------------|-------------|------------|--------------|
| preprocess (Python) | | | | |
| preprocess_cpp (C++) | | | | |

### Full pipeline (TRT model, Python postprocess)

| Pipeline | p50 @c=1 | p99 @c=1 | Peak throughput | Saturates at |
|----------|----------|----------|-----------------|--------------|
| yolov8s_trt_pipeline (Python pre) | 75ms | 79ms | 33 inf/s | c=3 |
| yolov8s_trt_pipeline_cpp (C++ pre) | | | | |

> Baseline numbers from experiment 01 (2 Python instances).

## Key takeaways

_Fill in after running._

## Hardware
- GPU: GTX 1060 3GB
- CPU: 4 cores / 8 threads
- Triton: 23.08, TensorRT 8.6
