import os
from dataclasses import dataclass

import yaml
import numpy as np
from PIL import Image

from .predictor import OnnxSessionWrapper, ImagePredictor
from ..utils.paths import get_metadata_path


@dataclass
class FeaturePrediction:
    feature: np.ndarray

    def to_dict(self):
        return {
            "feature": self.feature.tolist(),
        }


class OnnxFeatureExtractor(OnnxSessionWrapper):
    def __call__(self, pil_images: list[Image.Image]):
        images_array = self.collate_images(pil_images)
        input_dict = {self.input_names[0]: images_array}
        features = self.session.run(None, input_dict)[0]
        return features


class FeatureExtractionPredictor(ImagePredictor):
    model_type = "feature-extraction-onnx"
    backend_model = OnnxFeatureExtractor

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = self.backend_model(model_path, device)
        self.model_path = model_path

    def predict(self, images: list[Image.Image]) -> list[FeaturePrediction]:
        if not images:
            return []
        raw_outputs = self.model(images)
        outputs = []
        for feature in raw_outputs:
            outputs.append(FeaturePrediction(feature=feature))
        return outputs

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
