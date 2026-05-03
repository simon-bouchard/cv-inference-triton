#!/bin/bash
# Usage: bash benchmark_latency.sh <model_name>
# Example: bash benchmark_latency.sh yolov8s
#
# Run from inside the Triton SDK container with /benchmarks mounted.
# Tests one request at a time (concurrency=1) to measure raw latency.

MODEL=${1:?"Usage: $0 <model_name>"}
OUTPUT_DIR=/benchmarks/results
mkdir -p "$OUTPUT_DIR"

echo "=== Latency benchmark: $MODEL ==="

perf_analyzer \
  -m "$MODEL" \
  --input-data zero \
  --shape images:1,3,640,640 \
  -u localhost:8000 \
  -i http \
  --concurrency-range 1 \
  --measurement-interval 10000 \
  -f "$OUTPUT_DIR/${MODEL}_latency.csv"
