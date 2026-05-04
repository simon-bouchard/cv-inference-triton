#!/bin/bash
# Runs pipeline benchmarks for experiment 01 (ONNX vs TRT).
# Run from repo root on the host (not inside a container).
#
# Requirements: pip install tritonclient[http] tritonclient[grpc] numpy

SCRIPT_DIR="$(dirname "$0")"
RESULTS_DIR="$SCRIPT_DIR/results"
LOAD_TEST="benchmarks/load_test.py"
IMAGE="data/sample.jpg"
CONCURRENCY="1:8"
DURATION=30

mkdir -p "$RESULTS_DIR"

echo "=== ONNX pipeline ==="
python3 "$LOAD_TEST" \
  --model yolov8s_pipeline \
  --image "$IMAGE" \
  --protocol http \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --output "$RESULTS_DIR/yolov8s_pipeline_throughput.csv"

echo "=== TRT pipeline ==="
python3 "$LOAD_TEST" \
  --model yolov8s_trt_pipeline \
  --image "$IMAGE" \
  --protocol http \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --output "$RESULTS_DIR/yolov8s_trt_pipeline_throughput.csv"
