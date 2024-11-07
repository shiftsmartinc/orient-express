FROM ubuntu:22.04

RUN apt update && apt install -y pip

ADD requirements.txt .

RUN pip install -r requirements.txt

ADD model_package model_package
ADD run_inference_server.py .
ADD logging.conf .

ENTRYPOINT ["python3", "run_inference_server.py"]