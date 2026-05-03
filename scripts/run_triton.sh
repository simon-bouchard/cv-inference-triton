#!/bin/bash

docker run --rm \
  --gpus all \
  --shm-size=512m \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  --name triton \
  tritonserver-cv:23.08 \
  tritonserver --model-repository=/models
