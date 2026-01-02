FROM ubuntu:24.04
RUN apt update && apt install -y python3 python3-pip
RUN pip config set global.break-system-packages true
RUN pip install poetry
WORKDIR /app
ADD docker-image-onnx/ .
ADD poetry.lock .
ADD pyproject.toml .
ADD README.md .
ADD orient_express/ orient_express/
RUN poetry config virtualenvs.create false
RUN poetry install --with image-onnx
ENTRYPOINT ["python3", "run_inference_server.py"]
