#!/bin/bash
# Runs pipeline benchmarks for experiment 02 (C++ preprocess backend).
# Compares Python preprocess pipeline (from exp 01) against C++ preprocess pipeline.
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

echo "=== TRT pipeline — Python preprocess (baseline) ==="
python3 "$LOAD_TEST" \
  --model yolov8s_trt_pipeline \
  --image "$IMAGE" \
  --protocol http \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --output "$RESULTS_DIR/yolov8s_trt_pipeline_throughput.csv"

echo "=== TRT pipeline — C++ preprocess ==="
python3 "$LOAD_TEST" \
  --model yolov8s_trt_pipeline_cpp \
  --image "$IMAGE" \
  --protocol http \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --output "$RESULTS_DIR/yolov8s_trt_pipeline_cpp_throughput.csv"
