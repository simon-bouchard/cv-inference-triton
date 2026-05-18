#!/bin/bash
# Runs end-to-end pipeline benchmarks for experiment 04 (geoclassifier ONNX vs TRT).
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

echo "=== ONNX pipeline ==="
python3 "$LOAD_TEST" \
  --model geoclassifier_pipeline \
  --image "$IMAGE" \
  --protocol http \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --outputs label,confidence \
  --output "$RESULTS_DIR/geoclassifier_pipeline_throughput.csv"

echo "=== TRT pipeline ==="
python3 "$LOAD_TEST" \
  --model geoclassifier_trt_pipeline \
  --image "$IMAGE" \
  --protocol http \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --outputs label,confidence \
  --output "$RESULTS_DIR/geoclassifier_trt_pipeline_throughput.csv"
