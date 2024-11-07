FROM ubuntu:22.04

RUN apt update && apt install -y pip poetry

ADD docker-xgboost-scikit-learn/run_inference_server.py .
ADD poetry.lock .

ENTRYPOINT ["python3", "run_inference_server.py"]