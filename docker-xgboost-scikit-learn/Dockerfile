FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/python
WORKDIR /app

# Dependency layer: torch/torchvision resolve from the PyTorch CPU index via
# tool.uv.sources in pyproject.toml; onnxruntime-gpu (selected by platform
# marker) is skipped and replaced with CPU onnxruntime below.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev --group server \
    --no-install-package onnxruntime-gpu

COPY README.md ./
COPY orient_express/ orient_express/
RUN uv sync --frozen --no-dev --group server --no-install-package onnxruntime-gpu \
    && uv pip install "onnxruntime>=1.20,<2"

COPY docker-xgboost-scikit-learn/ ./
ENTRYPOINT ["/app/.venv/bin/python", "run_inference_server.py"]
