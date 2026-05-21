FROM ubuntu:24.04
RUN apt update && apt install -y python3 python3-pip
RUN pip config set global.break-system-packages true
RUN pip install poetry poetry-plugin-export
WORKDIR /app
COPY docker-xgboost-scikit-learn/ .
COPY poetry.lock .
COPY pyproject.toml .
COPY README.md .
COPY orient_express/ orient_express/
RUN poetry config virtualenvs.create false
RUN poetry export --without-hashes --only main,server -o requirements.txt
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        grep -v -E '^(nvidia-|triton==|torch==|torchvision==|xgboost==|onnxruntime-gpu==)' requirements.txt > requirements.cpu && \
        mv requirements.cpu requirements.txt && \
        TORCH_VERSION=$(grep -A1 '^name = "torch"$' poetry.lock | grep '^version' | sed 's/.*"\(.*\)".*/\1/') && \
        TORCHVISION_VERSION=$(grep -A1 '^name = "torchvision"$' poetry.lock | grep '^version' | sed 's/.*"\(.*\)".*/\1/') && \
        XGBOOST_VERSION=$(grep -A1 '^name = "xgboost"$' poetry.lock | grep '^version' | sed 's/.*"\(.*\)".*/\1/') && \
        ONNXRUNTIME_VERSION=$(grep -A1 '^name = "onnxruntime-gpu"$' poetry.lock | grep '^version' | sed 's/.*"\(.*\)".*/\1/') && \
        pip install --index-url https://download.pytorch.org/whl/cpu \
            torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION && \
        pip install --no-deps xgboost==$XGBOOST_VERSION && \
        pip install onnxruntime==$ONNXRUNTIME_VERSION; \
    fi
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "run_inference_server.py"]
