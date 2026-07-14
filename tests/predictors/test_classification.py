"""Classification predictor tests."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors.classification import (
    ClassificationPrediction,
    ClassificationPredictor,
)


class TestClassificationPredictor:
    """Tests for ClassificationPredictor and OnnxClassifier."""

    @pytest.fixture
    def mock_classifier_session(self, mock_onnx_session):
        """Creates a mock session configured for classification."""
        return mock_onnx_session(
            resolution=224,
            input_names=["images"],
            output_names=["scores"],
        )

    def test_empty_input(self, mock_classifier_session, class_mapping):
        """Predict with empty list returns empty list without calling session."""
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_classifier_session,
        ):
            predictor = ClassificationPredictor("fake.onnx", class_mapping)
            result = predictor.predict([])

            assert result == []
            assert len(mock_classifier_session.run_inputs) == 0

    def test_preprocessing_single_image(self, mock_classifier_session, class_mapping):
        """Single image is resized to model resolution."""
        img = Image.fromarray(np.zeros((100, 150, 3), dtype=np.uint8), mode="RGB")

        # Return scores for 3 classes
        mock_classifier_session.run_outputs = [np.array([[0.1, 0.7, 0.2]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_classifier_session,
        ):
            predictor = ClassificationPredictor("fake.onnx", class_mapping)
            predictor.predict([img])

            # Verify preprocessing
            assert len(mock_classifier_session.run_inputs) == 1
            input_dict = mock_classifier_session.run_inputs[0]
            images_array = input_dict["images"]

            assert images_array.shape == (1, 224, 224, 3)
            assert images_array.dtype == np.uint8

    def test_preprocessing_multiple_images(
        self, mock_classifier_session, sample_images, class_mapping
    ):
        """Multiple images of different sizes are all resized and batched."""
        mock_classifier_session.run_outputs = [
            np.array([[0.1, 0.7, 0.2], [0.1, 0.7, 0.2], [0.1, 0.7, 0.2]])
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_classifier_session,
        ):
            predictor = ClassificationPredictor("fake.onnx", class_mapping)
            predictor.predict(sample_images)

            input_dict = mock_classifier_session.run_inputs[0]
            images_array = input_dict["images"]

            assert images_array.shape == (3, 224, 224, 3)

    def test_postprocessing_class_selection(
        self, mock_classifier_session, class_mapping
    ):
        """Highest scoring class is selected correctly."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        # Class 2 (index 1) has highest score
        mock_classifier_session.run_outputs = [np.array([[0.1, 0.8, 0.1]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_classifier_session,
        ):
            predictor = ClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img])

            print(results)
            assert len(results) == 1
            assert results[0].clss == "dog"  # class 2
            assert results[0].score == pytest.approx(0.8)

    def test_postprocessing_class_scores_dict(
        self, mock_classifier_session, class_mapping
    ):
        """Class scores dictionary contains all classes with correct scores."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_classifier_session.run_outputs = [np.array([[0.2, 0.5, 0.3]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_classifier_session,
        ):
            predictor = ClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img])

            class_scores = results[0].class_scores
            assert class_scores["cat"] == pytest.approx(0.2)
            assert class_scores["dog"] == pytest.approx(0.5)
            assert class_scores["bird"] == pytest.approx(0.3)

    def test_prediction_to_dict(self, mock_classifier_session, class_mapping):
        """ClassificationPrediction.to_dict() produces expected structure."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_classifier_session.run_outputs = [np.array([[0.2, 0.5, 0.3]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_classifier_session,
        ):
            predictor = ClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img])

            result_dict = results[0].to_dict()
            assert "class" in result_dict
            assert "score" in result_dict
            assert "class_scores" in result_dict
            assert result_dict["class"] == "dog"

    def test_get_annotated_image_returns_none(
        self, mock_classifier_session, class_mapping
    ):
        """Classification predictor returns None for annotated image."""
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_classifier_session,
        ):
            predictor = ClassificationPredictor("fake.onnx", class_mapping)
            img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")
            pred = ClassificationPrediction(
                clss="cat", score=0.9, class_scores={"cat": 0.9}
            )

            result = predictor.get_annotated_image(img, pred)
            assert result is None
