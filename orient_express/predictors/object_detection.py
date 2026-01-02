import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms
import yaml

from .predictor import OnnxSessionWrapper, OnnxImagePredictor
from ..utils.image_processor import pil_to_opencv, opencv_to_pil
from ..utils.paths import get_metadata_path


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


class OnnxDetector(OnnxSessionWrapper):
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
        images_array, target_sizes_array = self.preprocess(pil_images)

        input_dict = {
            self.input_names[0]: images_array,
            self.input_names[1]: target_sizes_array,
        }

        boxes, scores, labels = self.session.run(None, input_dict)
        return self.postprocess(boxes, scores, labels, confidence)


class OnnxBoundingBoxPredictor(OnnxImagePredictor):
    model_type = "object-detection-onnx"
    backend_model = OnnxDetector

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
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            opencv_image = cv2.rectangle(opencv_image, (x1, y1), (x2, y2), color, 2)

        return opencv_to_pil(opencv_image)

    def dump(self, dir: str):
        metadata = {
            "model_type": self.model_type,
            "classes": self.classes,
            "model_file": os.path.basename(self.model_path),
        }
        metadata_path = get_metadata_path(dir)
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)
        # model is already saved in the model_path
        return [metadata_path, self.model_path]
