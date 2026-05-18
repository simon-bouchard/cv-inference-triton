# Experiment 01 — ONNX vs TensorRT

## What changed
- Converted `yolov8s` ONNX model to TensorRT FP16 using `trtexec`
- Created `yolov8s_trt` as a separate model in the repository alongside the original
- Created `yolov8s_trt_pipeline` ensemble routing through the TRT model
- 2 Python instances for both `preprocess` and `postprocess`

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
Input: 1280×720 JPEG, letterboxed to 640×640 (YOLOv8 standard)

## Results

### Model-only (GPU compute isolated)

| Model | p50 latency | p99 latency | GPU compute | Peak throughput | Saturates at |
|-------|-------------|-------------|-------------|-----------------|--------------|
| yolov8s (ONNX) | 26ms | 29ms | 16.0ms | 58.8 inf/s | c=2 |
| yolov8s_trt (TRT FP16) | 22ms | 28ms | 10.7ms | 88.0 inf/s | c=4 |

TRT FP16 gives a **33% reduction in GPU compute time** and **50% throughput gain** in isolation.

### Pipeline (2 Python preprocess/postprocess instances)

| Pipeline | p50 @c=1 | p99 @c=1 | Peak throughput | Saturates at |
|----------|----------|----------|-----------------|--------------|
| yolov8s_pipeline (ONNX) | 35.6ms | 36.4ms | 59.4 inf/s | c=3 |
| yolov8s_trt_pipeline (TRT) | 36.2ms | 43.6ms | 75.5 inf/s | c=5 |

TRT pipeline delivers **27% higher peak throughput** than ONNX. Unlike results with large images where Python GIL dominated completely, with 720p input the GPU improvement becomes visible in the end-to-end pipeline.

### Effect of preprocess instance count

Measured with a larger input image (not directly comparable to the 720p numbers above), doubling the Python preprocess instances from 1 to 2 roughly doubled preprocessing throughput and meaningfully reduced full pipeline latency. This confirmed the Python GIL as the bottleneck at the time — adding instances bypasses the GIL by running independent interpreter processes. The 2-instance configuration is used for all pipeline results above.

### gRPC vs HTTP

Tested gRPC on the ONNX pipeline at the same concurrency sweep. Peak throughput and latency were identical to HTTP (~33 inf/s at the time, pre-image-resize). Protocol overhead is negligible compared to preprocess and GPU compute time — HTTP is sufficient.

## Key takeaways

- TRT FP16 gives a real GPU speedup: 16ms → 10.7ms compute, 88 vs 58 inf/s peak
- The GPU advantage transfers to the pipeline with realistic input sizes (720p) — TRT peaks at 75 vs 59 inf/s for ONNX
- Python preprocess is still a partial bottleneck — the ONNX pipeline saturates at c=3 while TRT needs c=5, suggesting preprocess keeps up with ONNX but starts limiting TRT at higher concurrency
- Next bottleneck to address: replace Python preprocess with a C++ backend to let the TRT model run at its full capacity (see exp 02)

## Hardware
- GPU: GTX 1060 3GB
- CPU: 4 cores / 8 threads
- Triton: 23.08, TensorRT 8.6
