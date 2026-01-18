from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .predictor import OnnxSessionWrapper, ImagePredictor


@dataclass
class MultiLabelClassificationPrediction:
    classes: list[str]
    class_scores: dict[str, float]

    def to_dict(self):
        return {
            "classes": self.classes,
            "class_scores": self.class_scores,
        }


class OnnxMultiLabelClassifier(OnnxSessionWrapper):
    def __call__(self, pil_images: list[Image.Image]):
        images = [
            cv2.resize(np.array(pil_img), (self.resolution, self.resolution))
            for pil_img in pil_images
        ]
        images_array = np.array(images)

        input_dict = {self.input_names[0]: images_array}

        scores = self.session.run(None, input_dict)[0]
        return scores


class MultiLabelClassificationPredictor(ImagePredictor):
    model_type = "multi-label-classification-onnx"
    backend_model = OnnxMultiLabelClassifier

    def predict(self, images: list[Image.Image], confidence: float) -> list[MultiLabelClassificationPrediction]:
        if not images:
            return []
        raw_outputs = self.model(images)
        outputs = []
        for class_scores in raw_outputs:
            classes = []
            # self.classes is 1-indexed
            for idx, score in enumerate(class_scores):
                if score > confidence:
                    classes.append(self.classes.get(idx + 1, "Unknown"))
            outputs.append(
                MultiLabelClassificationPrediction(
                    classes=classes,
                    class_scores={
                        # self.classes is 1-indexed
                        clss: float(class_scores[class_idx - 1])
                        for class_idx, clss in self.classes.items()
                    },
                )
            )
        return outputs

    def get_annotated_image(
        self, image: Image.Image, predictions: MultiLabelClassificationPrediction
    ):
        return None
