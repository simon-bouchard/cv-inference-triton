# ── Stage 1: compile the C++ preprocess backend ──────────────────────────────
FROM nvcr.io/nvidia/tritonserver:23.08-py3 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libjpeg-turbo8-dev \
  && ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0 \
           /usr/lib/x86_64-linux-gnu/libturbojpeg.so \
  && rm -rf /var/lib/apt/lists/*

COPY backends/preprocess_cpp /build/preprocess_cpp
WORKDIR /build/preprocess_cpp
RUN cmake -DCMAKE_BUILD_TYPE=Release . && make -j$(nproc)

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM nvcr.io/nvidia/tritonserver:23.08-py3

# Python backend (preprocess, postprocess) still needs Pillow.
RUN pip install pillow

# libjpeg-turbo8 is the runtime-only package (no headers, much smaller).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-turbo8 \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/tritonserver/backends/preprocess_cpp
COPY --from=builder \
    /build/preprocess_cpp/libtriton_preprocess_cpp.so \
    /opt/tritonserver/backends/preprocess_cpp/
