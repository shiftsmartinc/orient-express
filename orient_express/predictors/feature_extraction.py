import os
from dataclasses import dataclass

import numpy as np
import yaml
from PIL import Image

from ..utils.paths import get_metadata_path
from .predictor import ImagePredictor


@dataclass
class FeaturePrediction:
    feature: np.ndarray

    def to_dict(self):
        return {
            "feature": self.feature.tolist(),
        }


class FeatureExtractionPredictor(ImagePredictor):
    model_type = "feature-extraction-onnx"

    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        super().__init__(model_path, classes=None, device=device, **kwargs)

    @classmethod
    def from_dir(cls, dir: str, metadata: dict, device: str = "cpu", **kwargs):
        if "model_file" not in metadata:
            raise Exception("No model_file defined in metadata.yaml")
        onnx_path = os.path.join(dir, metadata["model_file"])
        return cls(onnx_path, device, **kwargs)

    def predict(self, images: list[Image.Image]) -> list[FeaturePrediction]:
        if not images:
            return []
        feed = self.preprocess(images)
        return self.postprocess(self.infer(feed), feed)

    def postprocess(self, outputs, feed) -> list[FeaturePrediction]:
        return [FeaturePrediction(feature=feature) for feature in outputs[0]]

    def get_annotated_image(self, image: Image.Image, predictions: FeaturePrediction):
        return None

    def dump(self, dir: str):
        metadata = {
            "model_type": self.model_type,
            "model_file": os.path.basename(self.model_path),
        }
        metadata_path = get_metadata_path(dir)
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)
        # model is already saved in the model_path
        return [metadata_path, self.model_path]
