import logging
import os

from kserve import ModelServer
from pythonjsonlogger import jsonlogger


if __name__ == "__main__":
    logging.config.fileConfig("logging.conf")
    try:
        model_server = ModelServer(
            http_port=8080, workers=1, configure_logging=True, log_config="logging.conf"
        )
        model_server.start([])
    except Exception as e:
        logging.exception("Failed to start model")
        raise e