# Experiment 03 — Dynamic Batching

## What changed
- Added `yolov8s_dynamic`: same ONNX model re-exported with `dynamic=True opset=17`, enabling Triton's dynamic batcher (`max_batch_size: 8`, preferred batch sizes 4 and 8, 1ms queue delay)
- Tested model-only throughput to isolate the GPU batching effect from pipeline overhead

## Hypothesis
The GPU is now the bottleneck (exp 02). Grouping multiple concurrent requests into a single batched GPU call should increase throughput by amortising kernel launch overhead and better utilizing CUDA cores.

## How tests were run

**Model-only** (perf_analyzer, inside Triton SDK container):
```bash
bash benchmarks/03_dynamic_batching/benchmark.sh yolov8s
bash benchmarks/03_dynamic_batching/benchmark.sh yolov8s_dynamic
```

Concurrency sweep: 1–8 | Input: synthetic zeros

## Results

### Model-only throughput

| Model | p50 @c=1 | GPU compute @c=1 | Peak throughput | Saturates at |
|-------|----------|-------------------|-----------------|--------------|
| yolov8s (no batching) | 26.6ms | 16.0ms | 58.2 inf/s | c=2 |
| yolov8s_dynamic (batching) | 28.0ms | 17.1ms | 60.5 inf/s | c=6 |

Dynamic batching yields **no meaningful throughput gain** — peak throughput is virtually identical (~58 vs ~60 inf/s), and single-request latency is slightly higher due to the 1ms queue delay.

## Why batching didn't help

Dynamic batching improves GPU throughput when a single inference leaves CUDA cores idle — i.e., the GPU has spare parallelism that a larger batch can exploit. On the GTX 1060 (1152 CUDA cores), a single YOLOv8s inference at 640×640 already saturates the available compute. Batching N images simply creates N× the workload with no parallelism gain.

This is evident in the GPU compute time: 16ms at batch=1 scales roughly linearly with concurrency rather than staying flat, which would be the signature of a GPU with headroom.

Batching would show dramatic gains on larger GPUs (A100, V100) where a single inference uses a fraction of available CUDA cores.

## Key takeaways

- Dynamic batching provides no benefit on a compute-saturated small GPU
- The GTX 1060 is fully utilized at batch=1 — the GPU is the hard ceiling
- Further throughput gains require a faster GPU, not software optimization
- The complete bottleneck progression across experiments: Python GIL (exp 01) → C++ removed preprocess as bottleneck (exp 02) → GPU is the true ceiling (exp 03)

## Hardware
- GPU: GTX 1060 3GB
- CPU: 4 cores / 8 threads
- Triton: 23.08, TensorRT 8.6
