# cv-inference-triton

Learning project focused on deploying computer vision models with NVIDIA Triton Inference Server. The goal is to build practical experience with on-premise GPU model serving — model optimisation, ensemble pipelines, batching, and multi-model deployment on a single GPU.

## Hardware

- **Development:** Laptop running WSL2, used for writing code and running clients
- **Inference:** Desktop with GTX 1060 3GB running Ubuntu, accessed via SSH
- **Serving:** Docker + nvidia-container-toolkit, Triton 23.08

## Stack

- PyTorch → ONNX → TensorRT (model path)
- Triton Inference Server (serving)
- Python tritonclient HTTP/gRPC (clients)
- perf_analyzer + custom load tester (benchmarking)

## Model repository

| Model | Backend | Description |
|-------|---------|-------------|
| `yolov8s` | ONNX Runtime | YOLOv8s object detection |
| `yolov8s_trt` | TensorRT FP16 | TRT-optimised version |
| `preprocess` | Python | JPEG bytes → normalised tensor |
| `postprocess` | Python | Raw output → boxes/scores/class_ids with NMS |
| `yolov8s_pipeline` | Ensemble | preprocess → yolov8s → postprocess |
| `yolov8s_trt_pipeline` | Ensemble | preprocess → yolov8s_trt → postprocess |

## Benchmark summary

Each row is one configuration. Latency and throughput measured end-to-end from client perspective.

| Exp | Scope | Backend | Batching | Instances | p50 @c=1 | p99 @c=1 | Peak throughput | Saturates at |
|-----|-------|---------|----------|-----------|----------|----------|-----------------|--------------|
| 01 | model | ONNX | off | 1 | 26ms | 29ms | 58 inf/s | c=2 |
| 01 | model | TRT FP16 | off | 1 | 22ms | 28ms | 88 inf/s | c=3 |
| 01 | pipeline | ONNX | off | 1 | 78ms | 85ms | 17 inf/s | c=2 |
| 01 | pipeline | TRT FP16 | off | 1 | 73ms | 78ms | 17 inf/s | c=2 |
| 01 | pipeline | ONNX | off | 2 | 78ms | 90ms | 33 inf/s | c=3 |
| 01 | pipeline | TRT FP16 | off | 2 | 75ms | 79ms | 33 inf/s | c=3 |

> Full results and analysis in `benchmarks/01_onnx_vs_trt/notes.md`

## Experiments

| # | Topic | Status |
|---|-------|--------|
| 01 | ONNX vs TensorRT FP16 | ✅ Done |
| 02 | Dynamic batching | 🔜 Next |
| 03 | Multi-model deployment | 🔜 Planned |

## Benchmarking

Model-only tests use `perf_analyzer` from the Triton SDK container. Pipeline tests use a custom load tester that handles binary image input and supports both HTTP and gRPC.

```bash
# Model-only (inside SDK container)
bash benchmarks/01_onnx_vs_trt/benchmark.sh yolov8s_trt model

# Pipeline (from repo root)
bash benchmarks/01_onnx_vs_trt/run_pipeline.sh
```

See `benchmarks/load_test.py` for full usage options including gRPC and custom concurrency ranges.
