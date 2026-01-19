import os
import logging

import joblib
import yaml

from ..utils.paths import get_metadata_path

from .predictor import Predictor, ImagePredictor
from .instance_segmentation import (
    InstanceSegmentationPredictor,
    InstanceSegmentationPrediction,
)
from .object_detection import BoundingBoxPredictor, BoundingBoxPrediction
from .semantic_segmentation import (
    SemanticSegmentationPredictor,
    SemanticSegmentationPrediction,
)
from .classification import ClassificationPredictor, ClassificationPrediction
from .multi_label_classification import (
    MultiLabelClassificationPredictor,
    MultiLabelClassificationPrediction,
)


def get_predictor(dir: str, device: str = "cpu"):
    downloaded_files = os.listdir(dir)
    metadata_path = get_metadata_path(dir)
    if not os.path.exists(metadata_path):
        logging.warning(
            f"No metadata.yaml file found in {dir}. Will try to load model from joblib file."
        )
        for file in downloaded_files:
            if file.endswith(".joblib"):
                file_path = os.path.join(dir, file)
                return joblib.load(file_path)
        raise Exception(f"No joblib file found in {dir}")
    else:
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        model_type = metadata.get("model_type")
        if model_type is None:
            raise Exception("No model_type defined in metadata.yaml")
        if "model_file" not in metadata:
            raise Exception("No model_file defined in metadata.yaml")
        if model_type == "joblib":
            joblib_path = os.path.join(dir, metadata["model_file"])
            return joblib.load(joblib_path)
        elif model_type == InstanceSegmentationPredictor.model_type:
            return load_image_predictor(
                InstanceSegmentationPredictor, dir, metadata, device
            )
        elif model_type == BoundingBoxPredictor.model_type:
            return load_image_predictor(BoundingBoxPredictor, dir, metadata, device)
        elif model_type == SemanticSegmentationPredictor.model_type:
            return load_image_predictor(
                SemanticSegmentationPredictor, dir, metadata, device
            )
        elif model_type == ClassificationPredictor.model_type:
            return load_image_predictor(ClassificationPredictor, dir, metadata, device)
        elif model_type == MultiLabelClassificationPredictor.model_type:
            return load_image_predictor(
                MultiLabelClassificationPredictor, dir, metadata, device
            )
        else:
            raise Exception(f"Unknown model_type '{model_type}'")


def load_image_predictor(
    model_type: type[ImagePredictor], dir: str, metadata: dict, device: str = "cpu"
):
    onnx_path = os.path.join(dir, metadata["model_file"])
    if "classes" not in metadata:
        raise Exception("No classes defined in metadata.yaml")
    classes = metadata["classes"]
    return model_type(onnx_path, classes, device)
