from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..utils.image_processor import (
    array_to_base64_npy,
    mask_to_base64,
    opencv_to_pil,
    pil_to_opencv,
    resize_masks,
)
from .predictor import ImagePredictor


@dataclass
class SemanticSegmentationPrediction:
    class_mask: np.ndarray
    valid_mask: np.ndarray
    conf_masks: np.ndarray

    def to_dict(self, include_conf_masks: bool = False):
        dict_repr = {
            "class_mask": mask_to_base64(self.class_mask.astype(np.uint8)),
            "valid_mask": mask_to_base64(self.valid_mask.astype(np.uint8)),
        }
        if include_conf_masks:
            # float32 (C, H, W) — PNG can't hold floats, so ship lossless .npy
            dict_repr["conf_masks"] = array_to_base64_npy(self.conf_masks)
        return dict_repr


class SemanticSegmentationPredictor(ImagePredictor):
    model_type = "semantic-segmentation-onnx"

    def predict(
        self, images: list[Image.Image], confidence: float = 0.5
    ) -> list[SemanticSegmentationPrediction]:
        if not images:
            return []
        feed = self.preprocess(images)
        outputs = self.infer(feed)
        return self.postprocess(outputs, feed, confidence=confidence)

    def preprocess(self, images: list[Image.Image]):
        # target_sizes is postprocess context, not a model input; infer()
        # only feeds input_names to the session
        return {
            self.input_names[0]: self.collate_images(images),
            "target_sizes": self.collate_sizes(images),
        }

    def assemble_feed(self, arrays, sizes):
        return {
            self.input_names[0]: np.stack(arrays),
            "target_sizes": np.array(sizes, dtype=np.float32),
        }

    def postprocess(
        self, outputs, feed, confidence: float = 0.5
    ) -> list[SemanticSegmentationPrediction]:
        batch_masks = outputs[0]
        target_sizes = feed["target_sizes"]
        results = []
        for i in range(batch_masks.shape[0]):
            h, w = int(target_sizes[i][0]), int(target_sizes[i][1])
            masks = resize_masks(batch_masks[i], h, w)
            class_mask = np.argmax(masks, axis=0).astype(np.uint8)
            valid_mask = np.max(masks, axis=0) >= confidence
            results.append(
                SemanticSegmentationPrediction(
                    class_mask=class_mask,
                    valid_mask=valid_mask,
                    conf_masks=masks,
                )
            )
        return results

    def get_annotated_image(
        self,
        image: Image.Image,
        prediction: SemanticSegmentationPrediction,
        mask_opacity: float = 0.3,
    ) -> Image.Image:
        opencv_image = pil_to_opencv(image)
        mask_overlay = opencv_image.copy()

        for class_id, class_name in self.classes.items():
            fill_color = self.color_scheme.get(class_name, (120, 120, 120))
            mask_overlay[
                (prediction.class_mask == class_id) & prediction.valid_mask
            ] = fill_color[:3]

        annotated_image = cv2.addWeighted(
            mask_overlay, mask_opacity, opencv_image, 1 - mask_opacity, 0
        )

        return opencv_to_pil(annotated_image)
