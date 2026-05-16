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
Image classification and object detection are baseline — do not over-explain either. Project now covers both tasks across two deployed models.

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
├── benchmarks/          # perf_analyzer scripts and results per experiment
├── backends/            # custom Triton backends (preprocess_cpp C++ shared lib)
├── scripts/             # shell scripts: docker run, trtexec, perf_analyzer
└── data/                # sample images for testing clients
```

## Model Pipeline
PyTorch (.pt) → ONNX → Triton (ONNX Runtime backend) → TensorRT engine → Triton (TensorRT backend)

- Training is done on Kaggle, weights downloaded locally
- `.pt`, `.onnx`, and `.engine` files are gitignored — never commit binary weights
- ONNX export uses opset 17; dynamic axes off by default for TensorRT compatibility

## Current Models

### yolov8s — Object Detection
Trained on COCO128. Full preprocess → model → postprocess ensemble pipeline, with Python and C++ preprocess variants.
- Input: `images` [1, 3, 640, 640] FP32
- Output: `output0` [1, 84, 8400] FP32 (raw, pre-NMS)

### geoclassifier-v1 — Quebec Region Classification
Fine-tuned EfficientNet-V2-M (ImageNet pretrained) that classifies street-level photos into one of Quebec's 17 administrative regions. Trained on ~750 Mapillary images per region (notebook: `training/geoclassifier-1.ipynb`). Val accuracy: 90.8%, test accuracy: 88.8%.

Exported to ONNX FP16, opset 17. Weights go in `model_repository/geoclassifier/1/model.onnx` (gitignored, download from W&B).

- Input: `input` [1, 3, 480, 480] FP32
- Output: `output` [1, 17] FP32 (raw logits)
- Preprocessing (must match training exactly):
  1. Resize to 512×512 (bilinear)
  2. Center-crop to 480×480 (16px off each edge)
  3. Divide by 255, normalize with ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Class labels (index → region, alphabetical order):
  0 Abitibi-Temiscamingue, 1 Bas-Saint-Laurent, 2 Capitale-Nationale,
  3 Centre-du-Quebec, 4 Chaudiere-Appalaches, 5 Cote-Nord, 6 Estrie,
  7 Gaspesie-Iles-de-la-Madeleine, 8 Lanaudiere, 9 Laurentides, 10 Laval,
  11 Mauricie, 12 Monteregie, 13 Montreal, 14 Nord-du-Quebec,
  15 Outaouais, 16 Saguenay-Lac-Saint-Jean

## Triton Focus Areas
- Model repository structure and config.pbtxt configuration
- ONNX serving and TensorRT conversion (FP16)
- Custom C++ Triton backend (preprocess_cpp: JPEG decode + resize + normalize in one pass)
- Dynamic batching and microbatching
- Ensemble pipelines (preprocess → model → postprocess, and multi-model chaining)
- Performance benchmarking with perf_analyzer and custom load tester
- gRPC vs HTTP clients via tritonclient

## Experiments
| # | Topic | Status |
|---|-------|--------|
| 01 | ONNX vs TensorRT FP16 | Done |
| 02 | C++ preprocess backend | Done |
| 03 | Dynamic batching | In progress |
| 04 | Multi-model deployment (yolov8s + geoclassifier) | Planned |

## How to Help
- Be hands-on and specific — working configs, commands, and code over explanations
- 3GB VRAM is the serving constraint (not training) — factor this in for multi-model deployment
- Flag concepts that are most relevant for an on-premise CV deployment interview
- Assume Linux/Docker throughout — no Windows guidance
