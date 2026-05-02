# cv-inference-triton

## Goal
Learning project focused on training and deploying computer vision models using NVIDIA Triton Inference Server. Primary motivation is building practical experience with on-premise GPU model serving — specifically the kind of setup (Triton, microbatching, GPU deployment) required for a computer vision role.

## Hardware & Environment
- **Laptop (WSL2/Linux)**: development, writing code, running clients, lightweight work
- **Desktop (GTX 1060, 3GB VRAM, native Ubuntu)**: GPU-heavy work — Triton serving, TensorRT conversion
- **Kaggle (T4, 16GB VRAM)**: model training
- **Workflow**: SSH from laptop into desktop for GPU work; port-forward Triton HTTP/gRPC endpoints back to laptop
- **Tooling**: Docker + nvidia-container-toolkit on desktop; PyTorch, Ultralytics, tritonclient

## CV Experience Level
Image classification is baseline — do not over-explain it. Project focuses on object detection primarily, with segmentation as a possible extension.

## Project Structure
```
cv-inference-triton/
├── model_repository/    # Triton model configs and weights
│   └── <model_name>/
│       ├── config.pbtxt
│       └── 1/
│           └── model.onnx (or .engine)
├── clients/             # tritonclient inference scripts (HTTP and gRPC)
├── training/            # Kaggle notebooks (.ipynb)
├── scripts/             # shell scripts: docker run, trtexec, perf_analyzer
└── data/                # sample images for testing clients
```

## Model Pipeline
PyTorch (.pt) → ONNX → Triton (ONNX Runtime backend) → TensorRT engine → Triton (TensorRT backend)

- Training is done on Kaggle, weights downloaded locally
- `.pt`, `.onnx`, and `.engine` files are gitignored — never commit binary weights
- ONNX export uses opset 17, simplify=True, dynamic=False (static batch for TensorRT compatibility)

## Current Models
- **yolov8s**: trained on COCO128, exported to ONNX
  - Input: `images` [1, 3, 640, 640] FP32
  - Output: `output0` [1, 84, 8400] FP32 (raw, pre-NMS)

## Triton Focus Areas
- Model repository structure and config.pbtxt configuration
- ONNX serving and TensorRT conversion
- Dynamic batching and microbatching
- Ensemble pipelines (preprocess → model → postprocess, and multi-model chaining)
- Performance benchmarking with perf_analyzer
- gRPC vs HTTP clients via tritonclient

## How to Help
- Be hands-on and specific — working configs, commands, and code over explanations
- 3GB VRAM is the serving constraint (not training) — factor this in for multi-model deployment
- Flag concepts that are most relevant for an on-premise CV deployment interview
- Assume Linux/Docker throughout — no Windows guidance
