from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import yaml

from .utils import get_metadata_path


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


class OnnxClassificationPredictor:
    model_type = "classification-onnx"

    def __init__(self, model_path: str, classes: dict[int, str]):
        self.model_path = model_path
        self.model = OnnxClassifier(model_path)
        self.classes = classes

    def predict(self, images: list[Image]) -> list[ClassificationPrediction]:
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

    def get_annotated_image(
        self, image: Image, predictions: list[ClassificationPrediction]
    ):
        return None

    def dump(self, dir: str):
        metadata = {
            "model_type": self.model_type,
            "classes": self.classes,
        }
        metadata_path = get_metadata_path(dir)
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)
        # model is already saved in the model_path
        return [metadata_path, self.model_path]


class OnnxClassifier:
    def __init__(self, onnx_path, providers=["CPUExecutionProvider"]):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_reuse = True

        self.session = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=session_options
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        input_shape = self.session.get_inputs()[0].shape
        self.resolution = input_shape[1]

    def __call__(self, pil_images: list[Image]):
        images = [
            cv2.resize(np.array(pil_img), (self.resolution, self.resolution))
            for pil_img in pil_images
        ]
        images_tensor = np.array(images)

        input_dict = {self.input_names[0]: images_tensor}

        scores = self.session.run(None, input_dict)
        return scores
