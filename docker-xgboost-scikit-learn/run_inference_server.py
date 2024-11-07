import logging
import os

import gcsfs
import joblib
import pandas as pd
from kserve import Model, ModelServer


class ScikitLearnPipelineModel(Model):
    def __init__(self, name: str, artifacts_path: str):
        super().__init__(name)
        self.artifacts_path = artifacts_path
        self.model = None

    def load(self):
        model_path = os.path.join(self.artifacts_path, "model.joblib")
        logging.info(f"Loading model from {model_path}")

        # Create a GCSFileSystem object
        fs = gcsfs.GCSFileSystem()

        # Open the file from GCS and load the model
        with fs.open(model_path, "rb") as f:
            self.model = joblib.load(f)

        logging.info("Model loaded successfully")
        return self

    def predict(self, inputs):
        logging.info(inputs)
        print(inputs)
        return self.model.predict(inputs)


if __name__ == "__main__":
    logging.config.fileConfig("logging.conf")
    try:
        storage_uri = os.environ["AIP_STORAGE_URI"]

        model = ScikitLearnPipelineModel("orient-express-model", storage_uri)
        model.load()

        # Start the model server with the loaded model
        model_server = ModelServer(
            http_port=8080, workers=1, configure_logging=True, log_config="logging.conf"
        )
        model_server.start([model])
    except Exception as e:
        logging.exception("Failed to start model")
        raise e
