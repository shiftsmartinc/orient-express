import json
import logging
import logging.config
import os

import pandas as pd
from kserve import Model, ModelServer

from orient_express.vertex import download_artifacts, ARTIFACT_DIR
from orient_express.predictors import get_predictor


class ScikitLearnPipelineModel(Model):
    def __init__(self, name: str, artifacts_path: str):
        super().__init__(name)
        self.name = name
        self.artifacts_path = artifacts_path
        self.model = None

    def load(self):
        logging.info(f"[{self.name}] loading model from GCS")
        download_dir = os.path.join(ARTIFACT_DIR, self.name)
        download_artifacts(download_dir, self.artifacts_path)
        self.model = get_predictor(download_dir)
        self.ready = True
        logging.info(f"{self.name} loaded successfully")
        return self

    def predict(self, inputs, *args, **kwargs):
        logging.info(f"[{self.name}] executing prediction")
        decoded_input = self.decode_input(inputs)

        input_df = pd.DataFrame(decoded_input["instances"])

        predictions = self.model.predict(input_df)
        response = {"predictions": predictions.tolist()}
        return response

    def decode_input(self, input_data):
        logging.info(f"PayloadType: {type(input_data)}")
        if isinstance(input_data, (bytes, str)):
            return json.loads(input_data)
        elif isinstance(input_data, dict):
            return input_data
        else:
            raise Exception(f"Unsupported payload type {type(input_data)}")


if __name__ == "__main__":
    logging.config.fileConfig("logging.conf")

    try:
        storage_uri = os.environ["AIP_STORAGE_URI"]
        model_name = os.environ["MODEL_NAME"]

        model = ScikitLearnPipelineModel(model_name, storage_uri)
        model.load()

        model_server = ModelServer(http_port=8080, workers=1)
        model_server.start([model])
    except Exception as e:
        logging.exception("Failed to start model")
        raise e
