FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/python
WORKDIR /app

# Dependency layer. onnxruntime-gpu (selected by platform marker) is skipped
# and replaced with CPU onnxruntime below; nvidia-nccl (an xgboost transitive
# dep only needed for distributed training) is skipped to keep the image lean.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev --group server \
    --no-install-package onnxruntime-gpu --no-install-package nvidia-nccl-cu12

COPY README.md ./
COPY orient_express/ orient_express/
RUN uv sync --frozen --no-dev --group server --no-install-package onnxruntime-gpu --no-install-package nvidia-nccl-cu12 \
    && uv pip install "onnxruntime>=1.20,<2"

COPY docker-xgboost-scikit-learn/ ./
ENTRYPOINT ["/app/.venv/bin/python", "run_inference_server.py"]
