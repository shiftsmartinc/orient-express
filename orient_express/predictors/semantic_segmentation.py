from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .predictor import OnnxSessionWrapper, ImagePredictor
from ..utils.image_processor import pil_to_opencv, opencv_to_pil


@dataclass
class SemanticSegmentationPrediction:
    class_mask: np.ndarray
    conf_masks: np.ndarray

    def to_dict(self, include_conf_masks: bool = False):
        dict_repr = {
            "class_mask": self.class_mask,
        }
        if include_conf_masks:
            dict_repr["conf_masks"] = self.conf_masks.tolist()
        return dict_repr


class OnnxSemanticSegmentation(OnnxSessionWrapper):
    def preprocess(self, pil_images: list[Image.Image]):
        sizes = [[pil_img.size[1], pil_img.size[0]] for pil_img in pil_images]
        images = [
            cv2.resize(np.array(pil_img), (self.resolution, self.resolution))
            for pil_img in pil_images
        ]
        return np.array(images), np.array(sizes, dtype=np.float32)

    def postprocess(self, masks: np.ndarray, target_sizes: np.ndarray):
        results: list[np.ndarray] = []
        batch_size = masks.shape[0]

        for i in range(batch_size):
            filtered_masks = masks[i]

            h, w = int(target_sizes[i][0]), int(target_sizes[i][1])

            filtered_masks_torch = torch.from_numpy(filtered_masks).unsqueeze(1)
            resized_masks_torch = F.interpolate(
                filtered_masks_torch,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            resized_masks = resized_masks_torch.squeeze(1).numpy() > 0.0

            results.append(resized_masks)

        return results

    def __call__(self, pil_images: list[Image.Image]):
        images_array, target_sizes_array = self.preprocess(pil_images)
        input_dict = {self.input_names[0]: images_array}
        masks = self.session.run(None, input_dict)
        return self.postprocess(masks, target_sizes_array)


class SemanticSegmentationPredictor(ImagePredictor):
    model_type = "semantic-segmentation-onnx"
    backend_model = OnnxSemanticSegmentation

    def predict(
        self, images: list[Image.Image]
    ) -> list[SemanticSegmentationPrediction]:
        if not images:
            return []
        raw_outputs = self.model(images)
        outputs = []
        for masks in raw_outputs:
            class_mask = np.argmax(masks, axis=0)
            outputs.append(
                SemanticSegmentationPrediction(
                    class_mask=class_mask,
                    conf_masks=masks,
                )
            )
        return outputs

    def get_annotated_image(self, image: Image.Image, mask: np.array) -> Image.Image:
        opencv_image = pil_to_opencv(image)

        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

        for class_id, class_name in self.classes.items():
            fill_color = self.color_scheme.get(class_name, (120, 120, 120))
            mask_rgb[mask == class_id] = fill_color[:3]

        annotated_image = cv2.addWeighted(opencv_image, 0.7, mask_rgb, 0.3, 0)

        return opencv_to_pil(annotated_image)
