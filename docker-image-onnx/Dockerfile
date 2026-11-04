FROM debian:bookworm-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/python
WORKDIR /app

# The tensorrt wheels on PyPI are stubs that fetch the real wheel from
# pypi.nvidia.com with stdlib urllib, which reads the SYSTEM trust store
# rather than certifi — without this the install dies on
# CERTIFICATE_VERIFY_FAILED. Builder only: nothing here reaches the runtime
# stage, which still needs no CA bundle (see below).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer. The tensorrt12 extra ships one image for every runtime:
# GPU-less deployments run the bundled CPU EP (onnxruntime-gpu includes it;
# the CUDA user-space wheels satisfy its link without any driver), and GPU
# deployments pick cuda/tensorrt from the deployment's device= token. The
# CUDA-12 stack, not CUDA-13: Vertex prediction hosts run driver r535,
# below the r580 floor the cu13 wheels need. The `server` group is
# deliberately the lean one — pandas/scikit-learn/xgboost live in
# `server-tabular`, which only the joblib image needs, and they are what
# pulled in nvidia-nccl.
# The uv cache lives in a build-time mount so wheels never land in a layer.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --group server --extra tensorrt12

COPY README.md ./
COPY orient_express/ orient_express/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group server --extra tensorrt12

# The tensorrt wheel ships TensorRT's Windows builder resources alongside the
# Linux ones — 2.8GB of .so files this image can never load. TensorRT selects
# a builder resource by host arch, so removing the win_ set is inert here.
# The per-SM Linux resources stay: TRT needs the one matching the deployment
# GPU to compile an engine, and which GPU that is isn't known at build time.
RUN rm -f /app/.venv/lib/python*/site-packages/tensorrt_libs/libnvinfer_builder_resource_win_*

# Runtime stage: just the interpreter and the venv, no uv or build leftovers.
# No ca-certificates needed: all HTTPS goes through certifi-bundled clients.
FROM debian:bookworm-slim
COPY --from=builder /python /python
COPY --from=builder /app /app
WORKDIR /app

# Vertex injects the NVIDIA driver (libcuda + tools) at /usr/local/nvidia
# on GPU-attached deployments; the loader must be told to look there.
# Harmless when the path doesn't exist (CPU deployments).
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64

COPY docker-image-onnx/ ./
ENTRYPOINT ["/app/.venv/bin/python", "run_inference_server.py"]
