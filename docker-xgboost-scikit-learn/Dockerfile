FROM ubuntu:24.04
RUN apt update && apt install -y python3 python3-pip
RUN pip config set global.break-system-packages true
RUN pip install poetry
WORKDIR /app
COPY docker-xgboost-scikit-learn/ .
COPY poetry.lock .
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install --no-root
ENTRYPOINT ["poetry", "run", "python3", "run_inference_server.py"]
