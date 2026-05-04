#!/bin/bash
# Usage: bash benchmark.sh <type>
#
# Types:
#   preprocess   — latency (c=1) + throughput sweep (c=1:8) for both
#                  preprocess (Python) and preprocess_cpp (C++) in isolation
#   pipeline     — throughput sweep (c=1:8) comparing
#                  yolov8s_trt_pipeline vs yolov8s_trt_pipeline_cpp
#
# Run from inside the Triton SDK container:
#   docker run --rm -it --net=host \
#     -v $(pwd)/benchmarks:/benchmarks \
#     -v $(pwd)/data:/data \
#     nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
#
# Then: bash /benchmarks/02_cpp_backend/benchmark.sh <type>

TYPE=${1:?"Usage: $0 <type> (type: preprocess|pipeline)"}

OUTPUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUTPUT_DIR"

run_preprocess_bench() {
    local model=$1
    echo "=== Latency (c=1): $model ==="
    perf_analyzer \
      -m "$model" \
      --input-data /data/input \
      --shape image_raw:1 \
      -u localhost:8000 \
      -i http \
      --concurrency-range 1 \
      --measurement-interval 10000 \
      -f "$OUTPUT_DIR/${model}_latency.csv"

    echo "=== Throughput (c=1:8): $model ==="
    perf_analyzer \
      -m "$model" \
      --input-data /data/input \
      --shape image_raw:1 \
      -u localhost:8000 \
      -i http \
      --concurrency-range 1:8:1 \
      --measurement-interval 10000 \
      -f "$OUTPUT_DIR/${model}_throughput.csv"
}

if [ "$TYPE" = "preprocess" ]; then
    run_preprocess_bench preprocess
    run_preprocess_bench preprocess_cpp

elif [ "$TYPE" = "pipeline" ]; then
    for model in yolov8s_trt_pipeline yolov8s_trt_pipeline_cpp; do
        echo "=== Pipeline throughput (c=1:8): $model ==="
        perf_analyzer \
          -m "$model" \
          --input-data /data/input \
          --shape image_raw:1 \
          -u localhost:8000 \
          -i http \
          --concurrency-range 1:8:1 \
          --measurement-interval 10000 \
          -f "$OUTPUT_DIR/${model}_throughput.csv"
    done

else
    echo "Unknown type: $TYPE. Use 'preprocess' or 'pipeline'."
    exit 1
fi
