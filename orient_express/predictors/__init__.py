import logging
import os
from typing import Any, TypeVar, overload

import joblib
import yaml

from ..utils.paths import get_metadata_path
from .classification import ClassificationPrediction, ClassificationPredictor
from .feature_extraction import FeatureExtractionPredictor, FeaturePrediction
from .instance_segmentation import (
    InstanceSegmentationPrediction,
    InstanceSegmentationPredictor,
)
from .multi_label_classification import (
    MultiLabelClassificationPrediction,
    MultiLabelClassificationPredictor,
)
from .object_detection import BoundingBoxPrediction, BoundingBoxPredictor
from .predictor import PREDICTOR_REGISTRY, ImagePredictor, Predictor
from .semantic_segmentation import (
    SemanticSegmentationPrediction,
    SemanticSegmentationPredictor,
)
from .vector_index import CropSpec, SearchResult, VectorIndex, build_vector_index

T = TypeVar("T")


@overload
def get_predictor(dir: str, device: str = "cpu") -> Any: ...


@overload
def get_predictor(dir: str, device: str = "cpu", *, expected_type: type[T]) -> T: ...


def get_predictor(
    dir: str, device: str = "cpu", *, expected_type: type[T] | None = None
) -> Any:
    """Load whatever model artifact lives in `dir`.

    The concrete class is chosen at runtime by the model_type recorded in
    metadata.yaml, so the static return type is unknown. Pass expected_type
    to narrow the type for the checker and assert it at runtime:

        predictor = get_predictor(dir, expected_type=BoundingBoxPredictor)
    """
    metadata_path = get_metadata_path(dir)
    if not os.path.exists(metadata_path):
        logging.warning(
            f"No metadata.yaml file found in {dir}. Will try to load model from joblib file."
        )
        predictor = _load_joblib_fallback(dir)
    else:
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        model_type = metadata.get("model_type")
        if model_type is None:
            raise Exception("No model_type defined in metadata.yaml")
        if "model_file" not in metadata:
            raise Exception("No model_file defined in metadata.yaml")
        if model_type == "joblib":
            predictor = joblib.load(os.path.join(dir, metadata["model_file"]))
        else:
            predictor_class = PREDICTOR_REGISTRY.get(model_type)
            if predictor_class is None:
                raise Exception(f"Unknown model_type '{model_type}'")
            predictor = predictor_class.from_dir(dir, metadata, device)
    if expected_type is not None and not isinstance(predictor, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__} but artifact in {dir} loaded as "
            f"{type(predictor).__name__}"
        )
    return predictor


def _load_joblib_fallback(dir: str):
    for file in os.listdir(dir):
        if file.endswith(".joblib"):
            return joblib.load(os.path.join(dir, file))
    raise Exception(f"No joblib file found in {dir}")


def load_vector_index(dir: str, metadata: dict | None = None) -> VectorIndex:
    if metadata is None:
        with open(get_metadata_path(dir)) as f:
            metadata = yaml.safe_load(f)
    assert metadata is not None
    return VectorIndex.from_dir(dir, metadata)
