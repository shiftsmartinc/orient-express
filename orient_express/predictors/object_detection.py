from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torchvision.ops import nms
import yaml

from ..utils.image_processor import pil_to_opencv, opencv_to_pil
from ..utils.colors import generate_color_scheme
from .utils import get_metadata_path


@dataclass
class BoundingBoxPrediction:
    clss: str
    score: float
    bbox: np.ndarray  # x1, y1, x2, y2

    def to_dict(self):
        return {
            "class": self.clss,
            "score": self.score,
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
            },
        }


class OnnxBoundingBoxPredictor:
    model_type = "object-detection-onnx"

    def __init__(self, model_path: str, classes: dict[int, str]):
        self.model_path = model_path
        self.model = OnnxDetector(model_path)
        self.classes = classes
        self.color_scheme = generate_color_scheme(list(self.classes.values()))

    def predict(
        self, images: list[Image], confidence: float, nms_threshold: float | None = None
    ) -> list[list[BoundingBoxPrediction]]:
        raw_outputs = self.model(images, confidence)
        if nms_threshold is not None:
            nms_outputs = []
            for boxes in raw_outputs:
                nms_outputs.append(self.apply_nms(boxes, nms_threshold))
            raw_outputs = nms_outputs
        outputs = []
        for boxes in raw_outputs:
            outputs.append(self.format_output(boxes))
        return outputs

    def apply_nms(self, boxes: np.ndarray, iou_threshold: float):
        if not len(boxes):
            return boxes
        boxes_tensor = torch.from_numpy(boxes)
        indices = (
            nms(boxes_tensor[:, :4], boxes_tensor[:, 4], iou_threshold).cpu().numpy()
        )
        return boxes_tensor[indices].numpy()

    def format_output(self, boxes: np.ndarray):
        outputs: list[BoundingBoxPrediction] = []
        for box in boxes:
            outputs.append(
                BoundingBoxPrediction(
                    clss=self.classes.get(int(box[5]), "Unknown"),
                    score=float(box[4]),
                    bbox=box[:4],
                )
            )
        return outputs

    def get_annotated_image(
        self, image: Image, predictions: list[BoundingBoxPrediction]
    ):
        opencv_image = pil_to_opencv(image)

        for pred in predictions:
            color = self.color_scheme.get(pred.clss, (255, 255, 255))
            x1, y1, x2, y2 = pred.bbox
            opencv_image = cv2.rectangle(opencv_image, (x1, y1), (x2, y2), color, 2)

        return opencv_to_pil(opencv_image)

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


class OnnxDetector:
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

    def preprocess(self, pil_images: list[Image]):
        sizes = [[pil_img.size[1], pil_img.size[0]] for pil_img in pil_images]
        images = [
            cv2.resize(np.array(pil_img), (self.resolution, self.resolution))
            for pil_img in pil_images
        ]
        return np.array(images), np.array(sizes, dtype=np.float32)

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        confidence: float,
    ):
        results: list[np.ndarray] = []
        batch_size = boxes.shape[0]

        for i in range(batch_size):
            valid_mask = scores[i] > confidence

            filtered_boxes = boxes[i][valid_mask]
            filtered_scores = scores[i][valid_mask]
            filtered_labels = labels[i][valid_mask]

            result = np.column_stack(
                [
                    filtered_boxes,
                    filtered_scores.reshape(-1, 1),
                    filtered_labels.reshape(-1, 1),
                ]
            )

            results.append(result)

        return results

    def __call__(self, pil_images: list[Image], confidence: float = 0.5):
        images_tensor, target_sizes_tensor = self.preprocess(pil_images)

        input_dict = {
            self.input_names[0]: images_tensor,
            self.input_names[1]: target_sizes_tensor,
        }

        boxes, scores, labels = self.session.run(None, input_dict)
        return self.postprocess(boxes, scores, labels, confidence)
