#!/bin/bash
# Runs all experiment 02 benchmarks using load_test.py.
# Measures preprocess in isolation (Python vs C++) then full pipeline (Python vs C++ preprocess).
# Run from repo root on the host (not inside a container).
#
# Requirements: pip install tritonclient[http] numpy

SCRIPT_DIR="$(dirname "$0")"
RESULTS_DIR="$SCRIPT_DIR/results"
LOAD_TEST="benchmarks/load_test.py"
IMAGE="data/sample.jpg"
CONCURRENCY="1:8"
DURATION=30

mkdir -p "$RESULTS_DIR"

echo "=== Preprocess isolation — Python ==="
python3 "$LOAD_TEST" \
  --model preprocess \
  --image "$IMAGE" \
  --outputs images \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --output "$RESULTS_DIR/preprocess_throughput.csv"

echo "=== Preprocess isolation — C++ ==="
python3 "$LOAD_TEST" \
  --model preprocess_cpp \
  --image "$IMAGE" \
  --outputs images \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --output "$RESULTS_DIR/preprocess_cpp_throughput.csv"

echo "=== Full pipeline — Python preprocess (baseline from exp 01) ==="
python3 "$LOAD_TEST" \
  --model yolov8s_trt_pipeline \
  --image "$IMAGE" \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --output "$RESULTS_DIR/yolov8s_trt_pipeline_throughput.csv"

echo "=== Full pipeline — C++ preprocess ==="
python3 "$LOAD_TEST" \
  --model yolov8s_trt_pipeline_cpp \
  --image "$IMAGE" \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --output "$RESULTS_DIR/yolov8s_trt_pipeline_cpp_throughput.csv"
