from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from ..utils.image_processor import pil_to_opencv, opencv_to_pil
from ..utils.colors import generate_color_scheme
from .utils import get_metadata_path


@dataclass
class InstanceSegmentationPrediction:
    clss: str
    score: float
    bbox: np.ndarray  # x1, y1, x2, y2
    mask: np.ndarray

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
            "mask": self.mask,
        }


class OnnxSegmentationPredictor:
    model_type = "instance-segmentation-onnx"

    def __init__(self, model_path: str, classes: dict[int, str]):
        self.model_path = model_path
        self.model = OnnxInstanceSegmentation(model_path)
        self.classes = classes
        self.color_scheme = generate_color_scheme(list(self.classes.values()))

    def predict(
        self, images: list[Image], confidence: float
    ) -> list[list[InstanceSegmentationPrediction]]:
        if not images:
            return []
        raw_outputs = self.model(images, confidence)
        outputs = []
        for boxes, masks in raw_outputs:
            outputs.append(self.format_output(boxes, masks))
        return outputs

    def format_output(self, boxes: np.ndarray, masks: np.ndarray):
        outputs: list[InstanceSegmentationPrediction] = []
        for box, mask in zip(boxes, masks):
            outputs.append(
                InstanceSegmentationPrediction(
                    clss=self.classes.get(int(box[5]), "Unknown"),
                    score=float(box[4]),
                    bbox=box[:4],
                    mask=mask,
                )
            )
        return outputs

    def get_annotated_image(
        self, image: Image, predictions: list[InstanceSegmentationPrediction]
    ) -> Image:
        opencv_image = pil_to_opencv(image)
        mask_overlay = opencv_image.copy()

        for pred in predictions:
            fill_color = self.color_scheme.get(pred.clss, (120, 120, 120))
            mask_overlay[pred.mask] = fill_color[:3]

        alpha = 0.5
        annotated_image = cv2.addWeighted(
            mask_overlay, alpha, opencv_image, 1 - alpha, 0
        )

        class_counts = defaultdict(int)
        for pred in predictions:
            class_counts[pred.clss] += 1
            self.draw_mask_index(opencv_image, pred, class_counts[pred.clss])

        return opencv_to_pil(annotated_image)

    def draw_mask_index(
        self,
        opencv_image: np.ndarray,
        prediction: InstanceSegmentationPrediction,
        index: int,
    ):
        # Calculate the centroid of the bbox
        centroid_x = int((prediction.bbox[0] + prediction.bbox[2]) / 2)
        centroid_y = int((prediction.bbox[1] + prediction.bbox[3]) / 2)
        # Put the object count in the center of the bbox
        font_scale = 1.5
        font_thickness = 4
        text = str(index)
        text_size = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )[0]
        text_x = int(centroid_x - text_size[0] / 2)
        text_y = int(centroid_y + text_size[1] / 2)
        cv2.putText(
            opencv_image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

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


class OnnxInstanceSegmentation:
    def __init__(self, onnx_path, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

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
        masks: np.ndarray,
        target_sizes: np.ndarray,
        confidence: float,
    ):
        results: list[tuple[np.ndarray, np.ndarray]] = []
        batch_size = boxes.shape[0]

        for i in range(batch_size):
            valid_mask = scores[i] > confidence
            filtered_boxes = boxes[i][valid_mask]
            filtered_scores = scores[i][valid_mask]
            filtered_labels = labels[i][valid_mask]
            # masks are [num_valid, H_mask, W_mask]
            # H_mask and W_mask tend to be small, around 100 pixels, so we need
            # to resize them. Since the images in the batch can be of different
            # sizes, we can't resize them in the onnx graph.
            filtered_masks = masks[i][valid_mask]

            h, w = int(target_sizes[i][0]), int(target_sizes[i][1])

            if len(filtered_masks) > 0:
                filtered_masks_torch = torch.from_numpy(filtered_masks).unsqueeze(1)
                resized_masks_torch = F.interpolate(
                    filtered_masks_torch,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                resized_masks = resized_masks_torch.squeeze(1).numpy() > 0.0
            else:
                resized_masks = np.empty((0, h, w), dtype=bool)

            result = np.column_stack(
                [
                    filtered_boxes,
                    filtered_scores.reshape(-1, 1),
                    filtered_labels.reshape(-1, 1),
                ]
            )

            results.append((result, resized_masks))

        return results

    def __call__(self, pil_images: list[Image], confidence: float = 0.5):
        images_tensor, target_sizes_tensor = self.preprocess(pil_images)

        input_dict = {
            self.input_names[0]: images_tensor,
            self.input_names[1]: target_sizes_tensor,
        }

        boxes, scores, labels, masks = self.session.run(None, input_dict)
        return self.postprocess(
            boxes, scores, labels, masks, target_sizes_tensor, confidence
        )
