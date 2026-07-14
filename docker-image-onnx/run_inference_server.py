import logging
import logging.config
import os
from concurrent.futures import ThreadPoolExecutor

from kserve import Model, ModelServer

from orient_express.predictors import get_predictor
from orient_express.serving import build_predict_kwargs, decode_input, download_image
from orient_express.utils.image_processor import fix_rotation
from orient_express.vertex import ARTIFACT_DIR, download_artifacts


class OnnxImageModel(Model):
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

        assert self.model is not None

        try:
            decoded_input = decode_input(inputs)
            instances = decoded_input["instances"]
            parameters = decoded_input.get("parameters", {}) or {}
        except Exception as e:
            logging.exception(f"[{self.name}] failed to decode input: {e}\n{inputs}")
            return {"error": "Failed to decode input"}

        include_debug = bool(parameters.get("debug_image", True))
        predict_kwargs = build_predict_kwargs(self.model.predict, parameters)

        predictions: list[dict] = [{} for _ in instances]

        images = []
        image_idxs = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(download_image, instance["image"])
                for instance in instances
            ]

            for img_idx, future in enumerate(futures):
                try:
                    image = future.result()
                    image = fix_rotation(image)
                    images.append(image)
                    image_idxs.append(img_idx)
                except Exception as e:
                    logging.exception(
                        f"[{self.name}] failed to download image {img_idx}: {e}"
                    )
                    predictions[img_idx] = {"status": "failed to download image"}

        model_predictions = self.model.predict(images, **predict_kwargs)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.model.to_response, image, prediction, include_debug
                )
                for image, prediction in zip(images, model_predictions, strict=True)
            ]

            for pred_idx, future in enumerate(futures):
                img_idx = image_idxs[pred_idx]
                try:
                    predictions[img_idx] = future.result()
                except Exception as e:
                    logging.exception(
                        f"[{self.name}] failed to build response {img_idx}: {e}"
                    )
                    predictions[img_idx] = {"status": "failed to get debug image"}

        return {"predictions": predictions}


if __name__ == "__main__":
    logging.config.fileConfig("logging.conf")

    try:
        storage_uri = os.environ["AIP_STORAGE_URI"]
        model_name = os.environ["MODEL_NAME"]

        model = OnnxImageModel(model_name, storage_uri)
        model.load()

        model_server = ModelServer(http_port=8080, workers=1)
        model_server.start([model])
    except Exception as e:
        logging.exception("Failed to start model")
        raise e
