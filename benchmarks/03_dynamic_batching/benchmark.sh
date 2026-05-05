#!/bin/bash
# Usage: bash benchmark.sh <model_name>
#
# Benchmarks model-only throughput for dynamic batching comparison.
# Uses --shape without the batch dimension (Triton owns it for batched models).
#
# Examples:
#   bash benchmark.sh yolov8s          # baseline: no batching, batch dim in tensor
#   bash benchmark.sh yolov8s_dynamic  # dynamic batching enabled
#
# Run from inside the Triton SDK container:
#   docker run --rm -it --net=host \
#     -v $(pwd)/benchmarks:/benchmarks \
#     nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
#
# Then: bash /benchmarks/03_dynamic_batching/benchmark.sh <model_name>

MODEL=${1:?"Usage: $0 <model_name>"}

OUTPUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUTPUT_DIR"

# yolov8s has max_batch_size:0 so the batch dim is part of the tensor shape.
# yolov8s_dynamic has max_batch_size:8 so Triton owns the batch dim.
if [ "$MODEL" = "yolov8s" ]; then
    SHAPE="images:1,3,640,640"
else
    SHAPE="images:3,640,640"
fi

echo "=== Latency (concurrency=1): $MODEL ==="
perf_analyzer \
  -m "$MODEL" \
  --input-data zero \
  --shape "$SHAPE" \
  -u localhost:8000 \
  -i http \
  --concurrency-range 1 \
  --measurement-interval 10000 \
  -f "$OUTPUT_DIR/${MODEL}_latency.csv"

echo "=== Throughput (concurrency=1:8): $MODEL ==="
perf_analyzer \
  -m "$MODEL" \
  --input-data zero \
  --shape "$SHAPE" \
  -u localhost:8000 \
  -i http \
  --concurrency-range 1:8:1 \
  --measurement-interval 10000 \
  -f "$OUTPUT_DIR/${MODEL}_throughput.csv"
