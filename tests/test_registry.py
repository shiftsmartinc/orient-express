"""Tests for the predictor registry and typed loading."""

import os

import pytest
import yaml

from orient_express.predictors import (
    PREDICTOR_REGISTRY,
    BoundingBoxPredictor,
    ClassificationPredictor,
    FeatureExtractionPredictor,
    InstanceSegmentationPredictor,
    MultiLabelClassificationPredictor,
    SemanticSegmentationPredictor,
    VectorIndex,
    get_predictor,
)
from orient_express.predictors.predictor import Predictor


class TestRegistry:
    def test_all_model_types_registered(self):
        """The frozen model_type strings (persisted in GCS metadata) resolve."""
        expected = {
            "classification-onnx": ClassificationPredictor,
            "multi-label-classification-onnx": MultiLabelClassificationPredictor,
            "object-detection-onnx": BoundingBoxPredictor,
            "instance-segmentation-onnx": InstanceSegmentationPredictor,
            "semantic-segmentation-onnx": SemanticSegmentationPredictor,
            "feature-extraction-onnx": FeatureExtractionPredictor,
            "vector-index": VectorIndex,
        }
        for model_type, predictor_class in expected.items():
            assert PREDICTOR_REGISTRY[model_type] is predictor_class

    def test_duplicate_model_type_rejected(self):
        with pytest.raises(ValueError, match="already registered"):

            class Duplicate(Predictor):  # noqa: F841
                model_type = "vector-index"

    def test_subclass_without_model_type_not_registered(self):
        before = dict(PREDICTOR_REGISTRY)

        class Intermediate(Predictor):
            pass

        assert PREDICTOR_REGISTRY == before


class TestExpectedType:
    def test_mismatched_expected_type_raises(self, tmp_path):
        import numpy as np

        index = VectorIndex(vectors=np.ones((2, 4)), labels=["a", "b"])
        index.dump(str(tmp_path))

        with pytest.raises(TypeError, match="Expected BoundingBoxPredictor"):
            get_predictor(str(tmp_path), expected_type=BoundingBoxPredictor)

    def test_matching_expected_type_returns_predictor(self, tmp_path):
        import numpy as np

        index = VectorIndex(vectors=np.ones((2, 4)), labels=["a", "b"])
        index.dump(str(tmp_path))

        loaded = get_predictor(str(tmp_path), expected_type=VectorIndex)
        assert isinstance(loaded, VectorIndex)
        assert len(loaded) == 2

    def test_unknown_model_type_raises(self, tmp_path):
        with open(os.path.join(tmp_path, "metadata.yaml"), "w") as f:
            yaml.dump({"model_type": "not-a-thing", "model_file": "x.bin"}, f)
        with pytest.raises(Exception, match="Unknown model_type"):
            get_predictor(str(tmp_path))
