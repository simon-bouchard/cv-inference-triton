# cv-inference-triton

Portfolio project focused on deploying computer vision models with NVIDIA Triton Inference Server. The goal is to build practical experience with on-premise GPU inference serving — model optimisation, ensemble pipelines, C++ backends, and bottleneck analysis — emulating the kind of stack used in industrial CV deployments (e.g. mining, manufacturing).

The project follows a deliberate experimental arc: each experiment identifies a bottleneck, addresses it, and reveals the next one. The full progression is documented with real benchmark numbers.

## Hardware

- **Development:** Laptop running WSL2, used for writing code and running clients
- **Inference:** Desktop with GTX 1060 3GB running Ubuntu, accessed via SSH
- **Serving:** Docker + nvidia-container-toolkit, Triton 23.08

## Stack

- PyTorch → ONNX → TensorRT FP16 (model path)
- Triton Inference Server (serving)
- C++ custom backend (preprocess)
- Python tritonclient HTTP/gRPC (clients)
- perf_analyzer + custom load tester (benchmarking)

## Pipeline architecture

Requests arrive as raw JPEG bytes and flow through a Triton ensemble:

```
JPEG bytes → preprocess → yolov8s_trt → postprocess → boxes / scores / class_ids
```

**Preprocess** decodes JPEG, letterboxes to 640×640 (preserving aspect ratio, padding with grey 114/255 as YOLOv8 expects), normalises to [0,1], and converts HWC→CHW.  
**postprocess** applies NMS to the raw YOLOv8 output and returns bounding boxes, confidence scores, and class IDs.

The input format (JPEG over HTTP) reflects a realistic industrial camera scenario: IP cameras on a mining site typically stream MJPEG or H.264, and a capture service forwards individual frames to the inference server.

## Model repository

| Model | Backend | Description |
|-------|---------|-------------|
| `yolov8s` | ONNX Runtime | YOLOv8s object detection |
| `yolov8s_trt` | TensorRT FP16 | TRT-optimised version |
| `yolov8s_dynamic` | ONNX Runtime | Dynamic batch support (exp 03) |
| `preprocess` | Python | JPEG → letterbox → normalised CHW tensor |
| `preprocess_cpp` | C++ | Same pipeline, no Python overhead |
| `postprocess` | Python | Raw output → boxes/scores/class_ids with NMS |
| `yolov8s_pipeline` | Ensemble | preprocess → yolov8s → postprocess |
| `yolov8s_trt_pipeline` | Ensemble | preprocess → yolov8s_trt → postprocess |
| `yolov8s_trt_pipeline_cpp` | Ensemble | preprocess_cpp → yolov8s_trt → postprocess |

## Experiments

### Exp 01 — ONNX vs TensorRT FP16

**What:** Converted the YOLOv8s ONNX model to TensorRT FP16 using `trtexec` and compared GPU compute time and end-to-end pipeline throughput.

**Model-only results** (GPU compute isolated, synthetic input):

| Model | p50 latency | GPU compute | Peak throughput |
|-------|-------------|-------------|-----------------|
| yolov8s (ONNX) | 26ms | 16.0ms | 58 inf/s |
| yolov8s_trt (TRT FP16) | 22ms | 10.7ms | 88 inf/s |

TRT FP16 reduces GPU compute time by 33% and raises isolated throughput by 50%.

**Pipeline results** (1280×720 JPEG input, 2 preprocess/postprocess instances):

| Pipeline | p50 @c=1 | Peak throughput |
|----------|----------|-----------------|
| ONNX + Python preprocess | 35.6ms | 59 inf/s |
| TRT + Python preprocess | 36.2ms | 75 inf/s |

The TRT advantage is visible in the pipeline (75 vs 59 inf/s), but the GPU is not reaching its full 88 inf/s capacity. Profiling showed Python preprocess saturates at ~87 inf/s — it is the partial bottleneck that prevents the TRT model from running at its ceiling.

---

### Exp 02 — C++ Preprocess Backend

**What:** Replaced the Python preprocess model with a custom C++ Triton backend (a compiled shared library with no Python interpreter). The C++ backend performs the full JPEG decode → letterbox → normalize → HWC→CHW pipeline using libjpeg-turbo.

**Preprocess in isolation:**

| Backend | p50 @c=1 | Peak throughput |
|---------|----------|-----------------|
| Python | 22ms | 87 inf/s |
| C++ | 12.7ms | 114 inf/s |

C++ is 42% faster in latency and 31% higher in peak throughput. The key reason is the Python GIL: even with multiple instances, the GIL serialises interpreter operations. C++ has no such constraint.

**Postprocess was also measured in isolation:** 332 inf/s peak — 4× above the pipeline ceiling — so no C++ replacement is needed there.

**Full pipeline:**

| Pipeline | p50 @c=1 | Peak throughput | Saturates at |
|----------|----------|-----------------|--------------|
| TRT + Python preprocess | 36.2ms | 75 inf/s | c=5 |
| TRT + C++ preprocess | 25.0ms | 79 inf/s | c=3 |

C++ delivers 31% lower latency at c=1 and a higher peak throughput. At high concurrency both pipelines converge to the same ceiling because preprocess is no longer the bottleneck — the **TRT model (GPU) is**. The C++ pipeline reaches that ceiling at c=3 instead of c=5, meaning it saturates more efficiently with less queuing.

---

### Exp 03 — Dynamic Batching

**What:** Re-exported the ONNX model with a dynamic batch axis and enabled Triton's dynamic batcher (preferred batch sizes 4 and 8, 1ms queue delay). Tested model-only throughput to isolate the GPU batching effect.

**Model-only results:**

| Model | p50 @c=1 | Peak throughput |
|-------|----------|-----------------|
| yolov8s (no batching) | 26.6ms | 58 inf/s |
| yolov8s_dynamic (batching) | 28.0ms | 60 inf/s |

No meaningful gain. Peak throughput is virtually identical and single-request latency is slightly higher due to the queue delay.

**Why:** Dynamic batching helps when a single inference leaves CUDA cores idle — i.e. the GPU has spare parallelism that a larger batch can exploit. On the GTX 1060 (1152 CUDA cores), one YOLOv8s inference already saturates available compute. GPU compute time scales roughly linearly with concurrency rather than staying flat, which would be the signature of a GPU with headroom. On a larger GPU (A100, V100), the same experiment would show a 4–8× throughput increase.

---

## Bottleneck progression

```
Exp 01: ONNX pipeline:  59 inf/s,  35.6ms latency @c=1
        TRT pipeline:   75 inf/s,  36.2ms latency @c=1  (+27% throughput vs ONNX)
        GPU alone:      88 inf/s
        → TRT helps but Python GIL in preprocess prevents the GPU from reaching capacity

Exp 02: TRT + C++ preprocess:  79 inf/s,  25.0ms latency @c=1
        → Removing the GIL cuts latency by 31% and closes the gap to the GPU ceiling
        → Preprocess is no longer the bottleneck — GPU is

Exp 03: Dynamic batching on the GPU shows no gain
        → GPU is genuinely saturated at batch=1, not just appearing to be
        → Only a faster GPU can push throughput higher
```

From first pipeline to final: **+34% throughput** (59 → 79 inf/s) and **-30% latency** (35.6 → 25.0ms). The most expensive hardware resource (GPU) is now the limiting factor, not CPU software overhead.

## Benchmark summary

| Exp | Scope | Configuration | p50 @c=1 | Peak throughput | Saturates at |
|-----|-------|---------------|----------|-----------------|--------------|
| 01 | model | ONNX | 26ms | 58 inf/s | c=2 |
| 01 | model | TRT FP16 | 22ms | 88 inf/s | c=4 |
| 01 | pipeline | ONNX + Python preprocess ×2 | 35.6ms | 59 inf/s | c=3 |
| 01 | pipeline | TRT + Python preprocess ×2 | 36.2ms | 75 inf/s | c=5 |
| 02 | preprocess | Python (isolated) | 22ms | 87 inf/s | c=3 |
| 02 | preprocess | C++ (isolated) | 12.7ms | 114 inf/s | c=3 |
| 02 | pipeline | TRT + C++ preprocess | 25.0ms | 79 inf/s | c=3 |
| 02 | postprocess | Python (isolated) | 6.6ms | 332 inf/s | c=5 |
| 03 | model | ONNX, no batching | 26.6ms | 58 inf/s | c=2 |
| 03 | model | ONNX, dynamic batching | 28.0ms | 60 inf/s | c=6 |

Input: 1280×720 JPEG, letterboxed to 640×640. Latency and throughput measured end-to-end from the client.

> **gRPC vs HTTP:** tested in exp 01 — no measurable difference in throughput or latency. Protocol overhead is negligible compared to preprocess and GPU compute time.

## Benchmarking

Model-only tests use `perf_analyzer` from the Triton SDK container. Pipeline tests use a custom load tester (`benchmarks/load_test.py`) that handles binary JPEG input over HTTP or gRPC.

```bash
# Model-only (inside SDK container)
bash benchmarks/01_onnx_vs_trt/benchmark.sh yolov8s model
bash benchmarks/03_dynamic_batching/benchmark.sh yolov8s_dynamic

# Pipeline (from repo root on host)
bash benchmarks/01_onnx_vs_trt/run_pipeline.sh
bash benchmarks/02_cpp_backend/run_pipeline.sh
```

Full methodology, raw numbers, and analysis in each experiment's `notes.md`.
