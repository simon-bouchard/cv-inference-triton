#!/bin/bash
# Usage: bash benchmark.sh <type>
#
# Types:
#   postprocess — latency (concurrency=1) + throughput sweep (concurrency=1:8)
#                 for the postprocess model using synthetic input
#
# Examples:
#   bash benchmark.sh postprocess
#
# Run from inside the Triton SDK container:
#   docker run --rm -it --net=host \
#     -v $(pwd)/benchmarks:/benchmarks \
#     nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
#
# Then: bash /benchmarks/02_cpp_backend/benchmark.sh <type>

TYPE=${1:?"Usage: $0 <type> (type: postprocess)"}

OUTPUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUTPUT_DIR"

if [ "$TYPE" = "postprocess" ]; then
    echo "=== Latency benchmark (concurrency=1): postprocess ==="
    perf_analyzer \
      -m postprocess \
      --input-data zero \
      --shape output0:1,84,8400 \
      -u localhost:8000 \
      -i http \
      --concurrency-range 1 \
      --measurement-interval 10000 \
      -f "$OUTPUT_DIR/postprocess_latency.csv"

    echo "=== Throughput benchmark (concurrency=1:8): postprocess ==="
    perf_analyzer \
      -m postprocess \
      --input-data zero \
      --shape output0:1,84,8400 \
      -u localhost:8000 \
      -i http \
      --concurrency-range 1:8:1 \
      --measurement-interval 10000 \
      -f "$OUTPUT_DIR/postprocess_throughput.csv"

else
    echo "Unknown type: $TYPE. Use 'postprocess'."
    exit 1
fi
