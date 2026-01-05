from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .predictor import OnnxSessionWrapper, ImagePredictor


@dataclass
class ClassificationPrediction:
    clss: str
    score: float
    class_scores: dict[str, float]

    def to_dict(self):
        return {
            "class": self.clss,
            "score": self.score,
            "class_scores": self.class_scores,
        }


class OnnxClassifier(OnnxSessionWrapper):
    def __call__(self, pil_images: list[Image.Image]):
        images = [
            cv2.resize(np.array(pil_img), (self.resolution, self.resolution))
            for pil_img in pil_images
        ]
        images_array = np.array(images)

        input_dict = {self.input_names[0]: images_array}

        scores = self.session.run(None, input_dict)
        return scores


class ClassificationPredictor(ImagePredictor):
    model_type = "classification-onnx"
    backend_model = OnnxClassifier

    def predict(self, images: list[Image.Image]) -> list[ClassificationPrediction]:
        if not images:
            return []
        raw_outputs = self.model(images)
        outputs = []
        for class_scores in raw_outputs:
            max_class_idx = class_scores.argmax() + 1
            max_clss = self.classes.get(max_class_idx, "Unknown")
            outputs.append(
                ClassificationPrediction(
                    clss=max_clss,
                    score=float(class_scores[max_class_idx]),
                    class_scores={
                        clss: float(class_scores[class_idx - 1])
                        for class_idx, clss in self.classes.items()
                    },
                )
            )
        return outputs

    def get_annotated_image(self, image: Image.Image, predictions: ClassificationPrediction):
        return None
