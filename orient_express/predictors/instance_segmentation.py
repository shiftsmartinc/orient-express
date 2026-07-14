from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..utils.image_processor import opencv_to_pil, pil_to_opencv, resize_masks
from .predictor import ImagePredictor, OnnxSessionWrapper

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class InstanceSegmentationPrediction:
    clss: str
    score: float
    bbox: np.ndarray  # x1, y1, x2, y2 in original-image pixels
    # Raw model-resolution mask (float, unresized model output). Full-size
    # masks are ~2 MB each as booleans, so they are materialized on demand
    # via resized_mask(image) / InstanceSegmentationPredictor.resize_masks()
    # instead of at predict time.
    mask: np.ndarray

    def resized_mask(self, image: Image.Image) -> np.ndarray:
        """This prediction's mask resized to fit `image` (bool).

        It is the caller's responsibility to pass the image this prediction
        was made on.
        """
        width, height = image.size
        return resize_masks(self.mask[None].astype(np.float32), height, width)[0] > 0.0

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
            # raw model-resolution mask; resize via resized_mask(image) /
            # InstanceSegmentationPredictor.resize_masks(image, predictions)
            dict_repr["mask"] = self.mask.tolist()
        return dict_repr


class OnnxInstanceSegmentation(OnnxSessionWrapper):
    def __call__(self, pil_images: list[Image.Image], confidence: float = 0.5):
        images_array = self.collate_images(pil_images)
        target_sizes_array = self.collate_sizes(pil_images)

        input_dict = {
            self.input_names[0]: images_array,
            self.input_names[1]: target_sizes_array,
        }

        boxes, scores, labels, masks = self.session.run(None, input_dict)
        return self.postprocess(boxes, scores, labels, masks, confidence)

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        masks: np.ndarray,
        confidence: float,
    ):
        results: list[tuple[np.ndarray, np.ndarray]] = []
        batch_size = boxes.shape[0]

        for i in range(batch_size):
            valid_mask = scores[i] > confidence
            filtered_boxes = boxes[i][valid_mask]
            filtered_scores = scores[i][valid_mask]
            filtered_labels = labels[i][valid_mask]
            # masks stay at model resolution ([num_valid, H_mask, W_mask],
            # typically ~100x100); resizing to image size happens lazily via
            # resized_mask(image) / resize_masks(image, predictions).
            filtered_masks = masks[i][valid_mask]

            result = np.column_stack(
                [
                    filtered_boxes,
                    filtered_scores.reshape(-1, 1),
                    filtered_labels.reshape(-1, 1),
                ]
            )

            results.append((result, filtered_masks))

        return results


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
        for box, mask in zip(boxes, masks, strict=True):
            outputs.append(
                InstanceSegmentationPrediction(
                    clss=self.classes.get(int(box[5]), "Unknown"),
                    score=float(box[4]),
                    bbox=box[:4],
                    mask=mask,
                )
            )
        return outputs

    def resize_masks(
        self,
        image: Image.Image,
        predictions: list[InstanceSegmentationPrediction],
        index: int | None = None,
    ) -> np.ndarray:
        """Resize prediction masks to fit `image` (bool).

        Resizes all masks by default (returned as an (N, H, W) array, one
        batched threaded operation); pass index to resize a single
        prediction's mask (returned as (H, W)). It is the caller's
        responsibility to pass the image the predictions were made on.
        """
        if index is not None:
            return predictions[index].resized_mask(image)
        if not predictions:
            return np.empty((0, 0, 0), dtype=bool)
        width, height = image.size
        stacked = np.stack([pred.mask for pred in predictions]).astype(np.float32)
        return resize_masks(stacked, height, width) > 0.0

    def get_annotated_image(
        self,
        image: Image.Image,
        predictions: list[InstanceSegmentationPrediction],
        mask_opacity: float = 0.3,
        draw_indices: bool = True,
        font_scale: float | None = None,
    ) -> Image.Image:
        opencv_image = pil_to_opencv(image)

        if predictions:
            # Draw all model-resolution masks onto one model-resolution
            # canvas, then resize that single canvas to the image size —
            # instead of resizing every mask to full resolution.
            mask_h, mask_w = predictions[0].mask.shape
            overlay = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
            coverage = np.zeros((mask_h, mask_w), dtype=bool)
            for pred in predictions:
                fill_color = self.color_scheme.get(pred.clss, (120, 120, 120))
                mask = pred.mask > 0.0
                overlay[mask] = fill_color[:3]
                coverage |= mask

            height, width = opencv_image.shape[:2]
            overlay_full = cv2.resize(
                overlay, (width, height), interpolation=cv2.INTER_NEAREST
            )
            coverage_full = (
                cv2.resize(
                    coverage.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )

            blended = cv2.addWeighted(
                overlay_full, mask_opacity, opencv_image, 1 - mask_opacity, 0
            )
            annotated_image = np.where(coverage_full[..., None], blended, opencv_image)
        else:
            annotated_image = opencv_image

        if draw_indices:
            class_counts = defaultdict(int)
            for pred in predictions:
                class_counts[pred.clss] += 1
                self.draw_mask_index(
                    annotated_image,
                    pred,
                    class_counts[pred.clss],
                    font_scale,
                )

        return opencv_to_pil(annotated_image)

    def draw_mask_index(
        self,
        opencv_image: np.ndarray,
        prediction: InstanceSegmentationPrediction,
        index: int,
        font_scale: float | None = None,
    ):
        text = str(index)

        bbox = prediction.bbox
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        if font_scale is None:
            target_size = min(bbox_width, bbox_height) * 0.3
            (base_w, base_h), _ = cv2.getTextSize(text, FONT, 1, 1)
            font_scale = target_size / max(base_w, base_h)

        assert font_scale is not None
        thickness = max(int(font_scale * 2), 1)

        text_size = cv2.getTextSize(text, FONT, font_scale, thickness)[0]

        centroid_x = int((bbox[0] + bbox[2]) / 2)
        centroid_y = int((bbox[1] + bbox[3]) / 2)
        text_x = int(centroid_x - text_size[0] / 2)
        text_y = int(centroid_y + text_size[1] / 2)

        cv2.putText(
            opencv_image,
            text,
            (text_x, text_y),
            FONT,
            font_scale,
            (255, 255, 255),
            thickness,
        )
