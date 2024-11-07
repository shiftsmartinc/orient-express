FROM ubuntu:22.04

RUN apt update && apt install -y pip

RUN pip install poetry

ADD docker-xgboost-scikit-learn/run_inference_server.py .
ADD poetry.lock .
ADD pyproject.toml .

RUN poetry install --with inference

ENTRYPOINT ["poetry", "run", "python3", "run_inference_server.py"]