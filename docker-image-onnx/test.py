import os
import logging
import sys
import json
from pathlib import Path
import cv2
from PIL import Image
import gcsfs
import joblib
from typing import Any
import colorsys
from collections import defaultdict

import yaml
import cv2
import numpy as np
from PIL.Image import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
import base64
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ExifTags
import requests


import logging
import os
from urllib.parse import urlparse, unquote

from google.api_core.retry import Retry
from google.cloud import storage



class OnnxSegmentationPredictor:
    model_type = "instance-seg-onnx"

    def __init__(self, model_path: str, classes: dict[int, str], confidence: float):
        self.model_path = model_path
        self.model = OnnxInstanceSegmentation(model_path)
        self.classes = classes
        self.confidence = confidence
        self.image_processor = ImageProcessor()

    def format_output(self, boxes) -> list[dict]:
        outputs = []
        for box in boxes:
            klass = self.classes.get(int(box[5]), "Unknown")
            box_dict = {
                "class": klass,
                "score": float(box[4]),
                "bbox": {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3]),
                },
            }
            outputs.append(box_dict)
        return outputs

    def predict(self, image: Image) -> tuple:
        boxes, masks = self.model(image, self.confidence)
        outputs = self.format_output(boxes)
        debug_image = self.get_debug_image(image, masks, boxes)
        return outputs, debug_image

    def get_debug_image(
        self, image: Image, masks: np.ndarray, boxes: np.ndarray
    ) -> Image:
        opencv_image = self.image_processor.pil_to_opencv(image)
        mask_overlay = opencv_image.copy()
        unique_classes = np.unique(boxes[:, 5].astype(int)).tolist()
        color_scheme = generate_color_scheme(unique_classes)

        # Process predictions
        for mask_idx in range(masks.shape[0]):
            mask = masks[mask_idx]
            class_id = int(boxes[mask_idx][5])
            fill_color = color_scheme.get(class_id, (0, 255, 0))
            mask_overlay[mask] = fill_color[:3]

        alpha = 0.5
        debug_image = cv2.addWeighted(mask_overlay, alpha, opencv_image, 1 - alpha, 0)
        self.draw_mask_indices(debug_image, boxes)
        return self.image_processor.opencv_to_pil(debug_image)

    def draw_mask_indices(self, opencv_image: np.array, boxes: np.ndarray):
        class_counts = defaultdict(int)
        for box in boxes:
            class_id = int(box[5])
            prediction_class = self.classes.get(class_id, "Unknown")
            class_counts[prediction_class] += 1
            self.draw_mask_index(opencv_image, box, class_counts[prediction_class])

    def draw_mask_index(self, opencv_image: np.array, box: np.ndarray, index: int):
        # Calculate the centroid of the bbox
        centroid_x = int((box[0] + box[2]) / 2)
        centroid_y = int((box[1] + box[3]) / 2)
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
            "confidence": self.confidence,
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

    def preprocess(self, pil_images):
        sizes = [[pil_img.size[1], pil_img.size[0]] for pil_img in pil_images]
        images = [
            cv2.resize(np.array(pil_img), (self.resolution, self.resolution))
            for pil_img in pil_images
        ]
        return np.array(images), np.array(sizes, dtype=np.float32)

    def postprocess(
        self, boxes, scores, labels, masks, target_sizes, confidence_threshold
    ):
        results = []
        batch_size = boxes.shape[0]

        for i in range(batch_size):
            valid_mask = scores[i] > confidence_threshold
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

    def __call__(self, pil_images, confidence_threshold=0.5):
        single_image = False
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
            single_image = True
        images_tensor, target_sizes_tensor = self.preprocess(pil_images)
        input_dict = {
            self.input_names[0]: images_tensor,
            self.input_names[1]: target_sizes_tensor,
        }
        boxes, scores, labels, masks = self.session.run(self.output_names, input_dict)
        results = self.postprocess(
            boxes, scores, labels, masks, target_sizes_tensor, confidence_threshold
        )
        return results[0] if single_image else results



def get_default_retry_policy():
    return Retry(initial=1.0, maximum=10.0, multiplier=2.0)


def parse_gcs_url(gs_url):
    """
    Parse URL of a file, located in Google Cloud Storage bucket.
    """
    parsed = urlparse(gs_url)
    if not parsed.scheme == "gs" or not parsed.netloc:
        raise ValueError(f"Invalid GCS URL format for {gs_url}")
    bucket_name = parsed.netloc
    file_path = parsed.path.lstrip("/")  # Remove leading slash for consistency
    return bucket_name, file_path


def get_gcs_from_http_url(http_url):
    """
    If HTTP URL belongs to Google Storage Bucket, extract GCS URI, otherwise return None

    :param http_url:
    :return: gcs_uri
    """
    if not http_url.startswith("https://storage.googleapis.com/"):
        return None

    return unquote(http_url.replace("https://storage.googleapis.com/", "gs://"))


def exists(gs_url):
    """
    Check whether a GCS URL exists
    """
    bucket_name, path = parse_gcs_url(gs_url)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.exists()


def upload_file(
    file_path, gs_url, read_mode="rb", content_type="application/octet-stream"
):
    """
    Upload file to Google Cloud Storage

    Args:
        file_path (str): Path to the local file to upload
        gs_url (str): Google Cloud Storage URL where the file should be uploaded
        content_type (str): MIME type of the file being uploaded
    """
    bucket_name, gs_file_path = parse_gcs_url(gs_url)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gs_file_path)

    # Open file in read binary mode for uploading
    with open(file_path, read_mode) as file_obj:
        blob.upload_from_file(
            file_obj, content_type=content_type, retry=get_default_retry_policy()
        )

    logging.info(f"File uploaded to {gs_url}")


def download_file(gs_url, file_path):
    """
    Download file from Google Cloud Storage to local filesystem.

    Args:
        gs_url (str): Google Cloud Storage URL (e.g., 'gs://bucket-name/path/to/file.txt')
        file_path (str): Local file path where the file should be saved

    Raises:
        ValueError: If gs_url is not a valid GCS URL
        google.cloud.exceptions.NotFound: If the file doesn't exist in GCS
        IOError: If there's an issue writing to the local file path

    Example:
        download_file('gs://my-bucket/data/file.csv', '/local/path/file.csv')
    """
    bucket_name, gs_file_path = parse_gcs_url(gs_url)

    # Create local directory if it doesn't exist
    local_dir = os.path.dirname(file_path)
    if local_dir and not os.path.exists(local_dir):
        os.makedirs(local_dir)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gs_file_path)

    # Download the file
    blob.download_to_filename(file_path, retry=get_default_retry_policy())

    logging.info(f"File downloaded from {gs_url} to {file_path}")


def read_file_bytes(gs_url):
    """
    Read file, located in Google Storage bucket, as bytes
    """
    bucket_name, path = parse_gcs_url(gs_url)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.download_as_bytes()

class ImageProcessor:
    @classmethod
    def read_image_from_url(cls, http_url, http_as_gsc=False) -> Image:
        # Extract GSC URI from http link and download the file directly.
        # It will increase reliability, as it will use GCP driver to fetch data
        # If URL is not GCS HTTP URL, download it through HTTP
        if http_as_gsc:
            gs_uri = gs.get_gcs_from_http_url(http_url)
            if gs_uri:
                return cls.read_image_from_gs(gs_uri)

        response = requests.get(http_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image

    @classmethod
    def read_image_from_gs(cls, gs_url) -> Image:
        bytes_content = gs.read_file_bytes(gs_url)
        image = Image.open(BytesIO(bytes_content))
        return image

    @classmethod
    def get_orientation(cls, image):
        try:
            exif = image._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == "Orientation":
                        return value
        except (AttributeError, KeyError, IndexError):
            return None

    @classmethod
    def rotate_image(cls, image, orientation):
        if orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 6:
            image = image.rotate(270, expand=True)
        elif orientation == 8:
            image = image.rotate(90, expand=True)
        return image

    @classmethod
    def clean_exif(cls, image):
        if hasattr(image, "_getexif"):
            image.info.pop("exif", None)
            if hasattr(image, "_exif"):
                image._exif = None
        return image

    @classmethod
    def image_to_base64(cls, image):
        bytes_content = cls.image_to_bytes(image)
        return base64.b64encode(bytes_content).decode("utf-8")

    @classmethod
    def image_to_bytes(cls, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return buffered.getvalue()

    @classmethod
    def fix_rotation(cls, image):
        """
        Encode input image to base64, ensure it's rotated without EXIF tags
        """
        orientation = cls.get_orientation(image)
        if orientation:
            logging.info(f"Rotating image for orientation {orientation}")
            image = cls.rotate_image(image, orientation)

        return cls.clean_exif(image)

    @classmethod
    def normalize(cls, image, max_size=2048):
        image = cls.fix_rotation(image)
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image

    @classmethod
    def pil_to_opencv(cls, image: Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @classmethod
    def opencv_to_pil(cls, image: np.array):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)



def generate_color_scheme(
    strings: list[str],
    saturation: float = 0.7,
    lightness: float = 0.5,
    transparency: float = 0.6,
) -> dict[Any, tuple[int, int, int, int]]:
    """
    Assign unique colors to strings with max distance between them.
    """
    n = len(strings)
    colors = {}

    for i, string in enumerate(strings):
        hue = i / n  # Normalize hue to the range [0, 1]

        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        color_code = (
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255),
            int(transparency * 150),
        )
        colors[string] = color_code

    return colors

import os


def get_metadata_path(dir: str):
    metadata_path = os.path.join(dir, "metadata.yaml")
    return metadata_path


def main(image_filepath: str, artifacts_path: str):
    """
    Run segmentation model on a single image and save debug output.
    
    Args:
        image_filepath: Path to input image (local, GCS, or HTTP URL)
        artifacts_path: Path to model artifacts (GCS path)
    """
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    logging.info(f"Loading model from {artifacts_path}")
    model_path = f"{artifacts_path}/ckp-food-prep-segmentation.onnx"
    class_path = f"{artifacts_path}/ckp-food-prep-segmentation_classes.joblib"
    
    fs = gcsfs.GCSFileSystem()
    with fs.open(model_path, "rb") as f:
        onnx_bytes = f.read()
    with fs.open(class_path, "rb") as f:
        classes = joblib.load(f)
    
    model = OnnxSegmentationPredictor(onnx_bytes, classes, confidence=0.5)
    image_processor = ImageProcessor()
    
    # Load and process image
    logging.info(f"Loading image from {image_filepath}")
    if image_filepath.startswith("http"):
        image = image_processor.read_image_from_url(image_filepath)
    elif image_filepath.startswith("gs://"):
        image = image_processor.read_image_from_gs(image_filepath)
    else:
        # Local file
        image = Image.open(image_filepath)
    
    # Run prediction
    logging.info("Running prediction")
    image = image_processor.normalize(image)
    detections, debug_image = model.predict(image)
    
    # Save debug image
    input_path = Path(image_filepath)
    debug_output_path = input_path.parent / f"{input_path.stem}_debug{input_path.suffix}"
    
    logging.info(f"Saving debug image to {debug_output_path}")
    debug_image.save(str(debug_output_path))
    
    # Save detections as JSON
    detections_output_path = input_path.parent / f"{input_path.stem}_detections.json"
    logging.info(f"Saving detections to {detections_output_path}")
    with open(detections_output_path, "w") as f:
        json.dump(detections, f, indent=2)
    
    logging.info("Done!")
    print(f"\nDebug image saved to: {debug_output_path}")
    print(f"Detections saved to: {detections_output_path}")
    print(f"\nDetections summary: {len(detections)} objects detected")


if __name__ == "__main__":
    image_filepath = sys.argv[1]
    artifacts_path = os.environ["AIP_STORAGE_URI"]
    
    main(image_filepath, artifacts_path)
