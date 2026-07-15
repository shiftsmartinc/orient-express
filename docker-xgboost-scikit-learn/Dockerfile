FROM debian:bookworm-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/python
WORKDIR /app

# Dependency layer. onnxruntime-gpu (selected by platform marker) is skipped
# and replaced with CPU onnxruntime below; nvidia-nccl (an xgboost transitive
# dep only needed for distributed training) is skipped to keep the image lean.
# The uv cache lives in a build-time mount so wheels never land in a layer.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --group server \
    --no-install-package onnxruntime-gpu --no-install-package nvidia-nccl-cu12

COPY README.md ./
COPY orient_express/ orient_express/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group server --no-install-package onnxruntime-gpu --no-install-package nvidia-nccl-cu12 \
    && uv pip install "onnxruntime>=1.20,<2"

# Runtime stage: just the interpreter and the venv, no uv or build leftovers.
# No ca-certificates needed: all HTTPS goes through certifi-bundled clients.
FROM debian:bookworm-slim
COPY --from=builder /python /python
COPY --from=builder /app /app
WORKDIR /app

COPY docker-xgboost-scikit-learn/ ./
ENTRYPOINT ["/app/.venv/bin/python", "run_inference_server.py"]
