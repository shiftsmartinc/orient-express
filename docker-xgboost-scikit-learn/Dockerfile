FROM ubuntu:22.04

RUN apt update && apt install -y pip

RUN pip install poetry

COPY docker-xgboost-scikit-learn/run_inference_server.py .
COPY poetry.lock .
COPY pyproject.toml .
COPY docker-xgboost-scikit-learn/logging.conf .

RUN poetry install --with xgboost-scikit-learn --no-root

ENTRYPOINT ["poetry", "run", "python3", "run_inference_server.py"]
