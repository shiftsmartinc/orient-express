import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor

from kserve import ModelServer, Model

from orient_express.utils.image_processor import (
    read_image_from_url,
    read_image_from_gs,
    fix_rotation,
    image_to_base64,
    base64_to_image,
)
from orient_express.utils.retry import retry
from orient_express.vertex import download_artifacts, ARTIFACT_DIR
from orient_express.predictors import (
    get_predictor,
    ClassificationPredictor,
    SemanticSegmentationPredictor,
)


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
            decoded_input = self.decode_input(inputs)
            instances = decoded_input["instances"]
            parameters = decoded_input.get("parameters", {})
        except Exception as e:
            logging.exception(f"[{self.name}] failed to decode input: {e}\n{inputs}")
            return {"error": "Failed to decode input"}

        predictions: list[dict] = [{} for _ in instances]

        images = []
        image_idxs = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.download_image, instance["image"])
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

        if isinstance(self.model, ClassificationPredictor):
            model_predictions = self.model.predict(images)

            for pred_idx, prediction in enumerate(model_predictions):
                img_idx = image_idxs[pred_idx]
                predictions[img_idx] = prediction.to_dict()
                predictions[img_idx]["status"] = "success"

        else:
            if isinstance(self.model, SemanticSegmentationPredictor):
                model_predictions = self.model.predict(images)
            else:
                confidence = parameters.get("confidence", 0.5)
                model_predictions = self.model.predict(images, confidence)

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.get_debug_image, image, prediction)
                    for image, prediction in zip(images, model_predictions)
                ]

                for pred_idx, future in enumerate(futures):
                    img_idx = image_idxs[pred_idx]
                    try:
                        debug_b64 = future.result()
                        prediction = model_predictions[pred_idx]
                        if isinstance(prediction, list):
                            predictions_json = [pred.to_dict() for pred in prediction]
                        else:
                            predictions_json = prediction.to_dict()

                        predictions[img_idx] = {
                            "status": "success",
                            "predictions": predictions_json,
                            "debug_image": debug_b64,
                        }
                    except Exception as e:
                        logging.exception(
                            f"[{self.name}] failed to get debug image {img_idx}: {e}"
                        )
                        predictions[img_idx] = {"status": "failed to get debug image"}

        return {"predictions": predictions}

    def decode_input(self, input_data):
        logging.info(f"PayloadType: {type(input_data)}")
        if isinstance(input_data, (bytes, str)):
            return json.loads(input_data)
        elif isinstance(input_data, dict):
            return input_data
        else:
            raise Exception(f"unsupported payload type {type(input_data)}")

    @retry(retries=3)
    def download_image(self, image_address):
        if image_address.startswith("http"):
            # http url
            image = read_image_from_url(image_address)
        elif image_address.startswith("gs://"):
            # gs uri
            image = read_image_from_gs(image_address)
        elif image_address.startswith("data:"):
            # data URI format: data:image/png;base64,iVBORw0KGgo...
            base64_data = image_address.split(",", 1)[1]
            image = base64_to_image(base64_data)
        else:
            # raw base64
            image = base64_to_image(image_address)
        return image

    def get_debug_image(self, image, preds):
        assert self.model is not None
        debug_image = self.model.get_annotated_image(image, preds)
        if debug_image is None:  # classification model
            return None
        debug_image_b64 = image_to_base64(debug_image)
        return debug_image_b64


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
