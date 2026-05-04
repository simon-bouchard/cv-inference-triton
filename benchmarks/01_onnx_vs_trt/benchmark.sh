#!/bin/bash
# Usage: bash benchmark.sh <model_name> <type>
#
# Types:
#   model    — latency (concurrency=1) + throughput sweep (concurrency=1:8)
#   pipeline — throughput sweep only (concurrency=1:8), uses real image input
#
# Examples:
#   bash benchmark.sh yolov8s model
#   bash benchmark.sh yolov8s_trt model
#   bash benchmark.sh yolov8s_pipeline pipeline
#   bash benchmark.sh yolov8s_trt_pipeline pipeline
#
# Run from inside the Triton SDK container:
#   docker run --rm -it --net=host \
#     -v $(pwd)/benchmarks:/benchmarks \
#     -v $(pwd)/data:/data \
#     nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
#
# Then: bash /benchmarks/01_onnx_vs_trt/benchmark.sh <model_name> <type>

MODEL=${1:?"Usage: $0 <model_name> <type>"}
TYPE=${2:?"Usage: $0 <model_name> <type> (type: model|pipeline)"}

OUTPUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUTPUT_DIR"

if [ "$TYPE" = "model" ]; then
    echo "=== Latency benchmark (concurrency=1): $MODEL ==="
    perf_analyzer \
      -m "$MODEL" \
      --input-data zero \
      --shape images:1,3,640,640 \
      -u localhost:8000 \
      -i http \
      --concurrency-range 1 \
      --measurement-interval 10000 \
      -f "$OUTPUT_DIR/${MODEL}_model_latency.csv"

    echo "=== Throughput benchmark (concurrency=1:8): $MODEL ==="
    perf_analyzer \
      -m "$MODEL" \
      --input-data zero \
      --shape images:1,3,640,640 \
      -u localhost:8000 \
      -i http \
      --concurrency-range 1:8:1 \
      --measurement-interval 10000 \
      -f "$OUTPUT_DIR/${MODEL}_model_throughput.csv"

elif [ "$TYPE" = "pipeline" ]; then
    echo "=== Pipeline throughput benchmark (concurrency=1:8): $MODEL ==="
    perf_analyzer \
      -m "$MODEL" \
      --input-data /data/input \
      --shape image_raw:1 \
      -u localhost:8000 \
      -i http \
      --concurrency-range 1:8:1 \
      --measurement-interval 10000 \
      -f "$OUTPUT_DIR/${MODEL}_pipeline_throughput.csv"

else
    echo "Unknown type: $TYPE. Use 'model' or 'pipeline'."
    exit 1
fi
