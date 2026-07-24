from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..utils.image_processor import opencv_to_pil, pil_to_opencv
from .predictor import ImagePredictor


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression via cv2.dnn.NMSBoxes.

    Keeps boxes whose IoU with a higher-scoring kept box is <= iou_threshold
    (verified output-identical to torchvision.ops.nms). boxes are (N, 4) as
    x1, y1, x2, y2; returns kept indices ordered by descending score.
    """
    boxes_xywh = boxes.astype(np.float32, copy=True)
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]
    keep = cv2.dnn.NMSBoxes(
        boxes_xywh, scores.astype(np.float32), 0.0, float(iou_threshold)
    )
    return np.asarray(keep, dtype=np.int64).reshape(-1)


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


class BoundingBoxPredictor(ImagePredictor):
    model_type = "object-detection-onnx"

    def predict(
        self,
        images: list[Image.Image],
        confidence: float,
        nms_threshold: float | None = None,
    ) -> list[list[BoundingBoxPrediction]]:
        if not images:
            return []
        feed = self.preprocess(images)
        outputs = self.infer(feed)
        return self.postprocess(
            outputs, feed, confidence=confidence, nms_threshold=nms_threshold
        )

    def preprocess(self, images: list[Image.Image]):
        return {
            self.input_names[0]: self.collate_images(images),
            self.input_names[1]: self.collate_sizes(images),
        }

    def assemble_feed(self, arrays, sizes):
        return {
            self.input_names[0]: np.stack(arrays),
            self.input_names[1]: np.array(sizes, dtype=np.float32),
        }

    def postprocess(
        self,
        outputs,
        feed,
        confidence: float,
        nms_threshold: float | None = None,
    ) -> list[list[BoundingBoxPrediction]]:
        boxes, scores, labels = outputs
        results = []
        for i in range(boxes.shape[0]):
            valid_mask = scores[i] > confidence
            result = np.column_stack(
                [
                    boxes[i][valid_mask],
                    scores[i][valid_mask].reshape(-1, 1),
                    labels[i][valid_mask].reshape(-1, 1),
                ]
            )
            if nms_threshold is not None:
                result = self.apply_nms(result, nms_threshold)
            results.append(self.format_output(result))
        return results

    def apply_nms(self, boxes: np.ndarray, iou_threshold: float):
        if not len(boxes):
            return boxes
        indices = nms(boxes[:, :4], boxes[:, 4], iou_threshold)
        return boxes[indices]

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
        self,
        image: Image.Image,
        predictions: list[BoundingBoxPrediction],
        line_width: int = 8,
    ):
        opencv_image = pil_to_opencv(image)

        for pred in predictions:
            color = self.color_scheme.get(pred.clss, (255, 255, 255))
            x1, y1, x2, y2 = pred.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            opencv_image = cv2.rectangle(
                opencv_image, (x1, y1), (x2, y2), color, line_width
            )

        return opencv_to_pil(opencv_image)
