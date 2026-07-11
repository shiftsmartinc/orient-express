"""Multi-label classification predictor tests."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors.multi_label_classification import (
    MultiLabelClassificationPrediction,
    MultiLabelClassificationPredictor,
)


class TestMultiLabelClassificationPredictor:
    """Tests for MultiLabelClassificationPredictor and OnnxMultiLabelClassifier."""

    @pytest.fixture
    def mock_multi_label_session(self, mock_onnx_session):
        """Creates a mock session configured for multi-label classification."""
        return mock_onnx_session(
            resolution=224,
            input_names=["images"],
            output_names=["scores"],
        )

    def test_empty_input(self, mock_multi_label_session, class_mapping):
        """Predict with empty list returns empty list without calling session."""
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            result = predictor.predict([], confidence=0.5)

            assert result == []
            assert len(mock_multi_label_session.run_inputs) == 0

    def test_preprocessing_single_image(self, mock_multi_label_session, class_mapping):
        """Single image is resized to model resolution."""
        img = Image.fromarray(np.zeros((100, 150, 3), dtype=np.uint8), mode="RGB")

        mock_multi_label_session.run_outputs = [np.array([[0.1, 0.7, 0.2]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            predictor.predict([img], confidence=0.5)

            assert len(mock_multi_label_session.run_inputs) == 1
            input_dict = mock_multi_label_session.run_inputs[0]
            images_array = input_dict["images"]

            assert images_array.shape == (1, 224, 224, 3)
            assert images_array.dtype == np.uint8

    def test_preprocessing_multiple_images(
        self, mock_multi_label_session, sample_images, class_mapping
    ):
        """Multiple images of different sizes are all resized and batched."""
        mock_multi_label_session.run_outputs = [
            np.array([[0.1, 0.7, 0.2], [0.8, 0.3, 0.9], [0.4, 0.4, 0.4]])
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            predictor.predict(sample_images, confidence=0.5)

            input_dict = mock_multi_label_session.run_inputs[0]
            images_array = input_dict["images"]

            assert images_array.shape == (3, 224, 224, 3)

    def test_postprocessing_confidence_filtering(
        self, mock_multi_label_session, class_mapping
    ):
        """Only classes with scores above threshold appear in classes list."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        # Scores: cat=0.3, dog=0.7, bird=0.6
        mock_multi_label_session.run_outputs = [np.array([[0.3, 0.7, 0.6]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert len(results) == 1
            assert set(results[0].classes) == {"dog", "bird"}
            assert "cat" not in results[0].classes

    def test_postprocessing_class_scores_dict_contains_all_classes(
        self, mock_multi_label_session, class_mapping
    ):
        """class_scores dict contains all classes with correct scores regardless of threshold."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_multi_label_session.run_outputs = [np.array([[0.2, 0.8, 0.5]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.9)  # High threshold

            class_scores = results[0].class_scores
            assert class_scores["cat"] == pytest.approx(0.2)
            assert class_scores["dog"] == pytest.approx(0.8)
            assert class_scores["bird"] == pytest.approx(0.5)

    def test_postprocessing_no_classes_pass_threshold(
        self, mock_multi_label_session, class_mapping
    ):
        """When no class passes threshold, classes list is empty but class_scores populated."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_multi_label_session.run_outputs = [np.array([[0.1, 0.2, 0.3]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert results[0].classes == []
            assert len(results[0].class_scores) == 3

    def test_postprocessing_all_classes_pass_threshold(
        self, mock_multi_label_session, class_mapping
    ):
        """When all classes exceed threshold, all appear in classes list."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_multi_label_session.run_outputs = [np.array([[0.9, 0.8, 0.7]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert set(results[0].classes) == {"cat", "dog", "bird"}

    def test_postprocessing_multiple_images_independent_filtering(
        self, mock_multi_label_session, class_mapping
    ):
        """Each image's predictions are filtered independently."""
        img1 = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")
        img2 = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_multi_label_session.run_outputs = [
            np.array(
                [
                    [0.9, 0.1, 0.1],  # Image 1: only cat
                    [0.1, 0.9, 0.9],  # Image 2: dog and bird
                ]
            )
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img1, img2], confidence=0.5)

            assert len(results) == 2
            assert results[0].classes == ["cat"]
            assert set(results[1].classes) == {"dog", "bird"}

    def test_postprocessing_unknown_class_handling(self, mock_multi_label_session):
        """Class indices not in mapping are labeled as Unknown."""
        # Sparse class mapping: only index 1 and 3 defined
        sparse_mapping = {1: "cat", 3: "bird"}
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        # Three scores, but index 2 (second score) has no mapping
        mock_multi_label_session.run_outputs = [np.array([[0.9, 0.9, 0.9]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", sparse_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert "cat" in results[0].classes
            assert "bird" in results[0].classes
            assert "Unknown" in results[0].classes

    def test_prediction_to_dict(self, mock_multi_label_session, class_mapping):
        """MultiLabelClassificationPrediction.to_dict() produces expected structure."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_multi_label_session.run_outputs = [np.array([[0.2, 0.8, 0.6]])]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            result_dict = results[0].to_dict()
            assert "classes" in result_dict
            assert "class_scores" in result_dict
            assert set(result_dict["classes"]) == {"dog", "bird"}
            assert result_dict["class_scores"]["cat"] == pytest.approx(0.2)

    def test_get_annotated_image_returns_none(
        self, mock_multi_label_session, class_mapping
    ):
        """Multi-label classification predictor returns None for annotated image."""
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_multi_label_session,
        ):
            predictor = MultiLabelClassificationPredictor("fake.onnx", class_mapping)
            img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")
            pred = MultiLabelClassificationPrediction(
                classes=["cat", "dog"],
                class_scores={"cat": 0.9, "dog": 0.8, "bird": 0.1},
            )

            result = predictor.get_annotated_image(img, pred)
            assert result is None
