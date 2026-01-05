from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .predictor import OnnxSessionWrapper, ImagePredictor
from ..utils.image_processor import pil_to_opencv, opencv_to_pil


@dataclass
class InstanceSegmentationPrediction:
    clss: str
    score: float
    bbox: np.ndarray  # x1, y1, x2, y2
    mask: np.ndarray

    def to_dict(self, include_mask: bool = False):
        dict_repr = {
            "class": self.clss,
            "score": self.score,
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
            },
        }
        if include_mask:
            dict_repr["mask"] = self.mask.tolist()
        return dict_repr


class OnnxInstanceSegmentation(OnnxSessionWrapper):
    def preprocess(self, pil_images: list[Image.Image]):
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

    def __call__(self, pil_images: list[Image.Image], confidence: float = 0.5):
        images_array, target_sizes_array = self.preprocess(pil_images)

        input_dict = {
            self.input_names[0]: images_array,
            self.input_names[1]: target_sizes_array,
        }

        boxes, scores, labels, masks = self.session.run(None, input_dict)
        return self.postprocess(
            boxes, scores, labels, masks, target_sizes_array, confidence
        )


class InstanceSegmentationPredictor(ImagePredictor):
    model_type = "instance-segmentation-onnx"
    backend_model = OnnxInstanceSegmentation

    def predict(
        self, images: list[Image.Image], confidence: float
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
        self, image: Image.Image, predictions: list[InstanceSegmentationPrediction]
    ) -> Image.Image:
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
            self.draw_mask_index(annotated_image, pred, class_counts[pred.clss])

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
