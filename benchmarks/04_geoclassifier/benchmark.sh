#!/bin/bash
# Usage: bash benchmark.sh <model_name>
#
# Examples:
#   bash benchmark.sh geoclassifier
#   bash benchmark.sh geoclassifier_trt
#
# Run from inside the Triton SDK container:
#   docker run --rm -it --net=host \
#     -v $(pwd)/benchmarks:/benchmarks \
#     nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
#
# Then: bash /benchmarks/04_geoclassifier/benchmark.sh <model_name>

MODEL=${1:?"Usage: $0 <model_name>"}

OUTPUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUTPUT_DIR"

echo "=== Latency benchmark (concurrency=1): $MODEL ==="
perf_analyzer \
  -m "$MODEL" \
  --input-data zero \
  --shape input:1,3,480,480 \
  -u localhost:8000 \
  -i http \
  --concurrency-range 1 \
  --measurement-interval 10000 \
  -f "$OUTPUT_DIR/${MODEL}_latency.csv"

echo "=== Throughput benchmark (concurrency=1:8): $MODEL ==="
perf_analyzer \
  -m "$MODEL" \
  --input-data zero \
  --shape input:1,3,480,480 \
  -u localhost:8000 \
  -i http \
  --concurrency-range 1:8:1 \
  --measurement-interval 10000 \
  -f "$OUTPUT_DIR/${MODEL}_throughput.csv"
