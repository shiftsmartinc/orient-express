"""
Tests for predictor classes.

These tests verify that:
1. Preprocessing produces correctly shaped and ordered arrays
2. Target sizes contain original image dimensions
3. Postprocessing correctly filters, formats, and resizes outputs
4. Dataclass formatting produces expected structures
5. Annotation methods modify images correctly in expected regions
"""

import os

import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from orient_express.predictors import load_vector_index
from orient_express.predictors.classification import (
    ClassificationPredictor,
    ClassificationPrediction,
)
from orient_express.predictors.multi_label_classification import (
    MultiLabelClassificationPredictor,
    MultiLabelClassificationPrediction,
)
from orient_express.predictors.object_detection import (
    BoundingBoxPredictor,
    BoundingBoxPrediction,
)
from orient_express.predictors.instance_segmentation import (
    InstanceSegmentationPredictor,
    InstanceSegmentationPrediction,
)
from orient_express.predictors.semantic_segmentation import (
    SemanticSegmentationPredictor,
    SemanticSegmentationPrediction,
)
from orient_express.predictors.vector_index import (
    VectorIndex,
    SearchResult,
    CropSpec,
    build_vector_index,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_onnx_session():
    """
    Creates a mock ONNX session factory that captures inputs and returns
    configured outputs.

    Usage: After creating the mock, set mock_session.run_outputs to the list
    that session.run() should return. The inputs will be captured in
    mock_session.run_inputs.
    """

    def _create_mock(resolution, input_names, output_names):
        mock_session = MagicMock()

        # Create mock inputs with proper .name attribute
        mock_inputs = []
        for inp_name in input_names:
            mock_input = MagicMock()
            mock_input.name = inp_name
            mock_input.shape = [None, resolution, resolution, 3]
            mock_inputs.append(mock_input)
        mock_session.get_inputs.return_value = mock_inputs

        # Create mock outputs with proper .name attribute
        mock_outputs = []
        for out_name in output_names:
            mock_output = MagicMock()
            mock_output.name = out_name
            mock_outputs.append(mock_output)
        mock_session.get_outputs.return_value = mock_outputs

        # Storage for captured inputs and configured outputs
        mock_session.run_inputs = []
        mock_session.run_outputs = []

        def capture_and_return(output_names, input_dict):
            mock_session.run_inputs.append(input_dict)
            return mock_session.run_outputs

        mock_session.run.side_effect = capture_and_return
        return mock_session

    return _create_mock


@pytest.fixture
def checkerboard_image():
    """
    Creates a 100x100 checkerboard image with 50x50 quadrants.
    Top-left: white (255, 255, 255)
    Top-right: black (0, 0, 0)
    Bottom-left: black (0, 0, 0)
    Bottom-right: white (255, 255, 255)
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0:50, 0:50] = [255, 255, 255]  # top-left white
    img[0:50, 50:100] = [0, 0, 0]  # top-right black
    img[50:100, 0:50] = [0, 0, 0]  # bottom-left black
    img[50:100, 50:100] = [255, 255, 255]  # bottom-right white
    return Image.fromarray(img, mode="RGB")


@pytest.fixture
def sample_images():
    """Creates a list of sample images with different sizes."""
    img1 = Image.fromarray(
        np.full((100, 150, 3), [255, 0, 0], dtype=np.uint8), mode="RGB"
    )
    img2 = Image.fromarray(
        np.full((200, 100, 3), [0, 255, 0], dtype=np.uint8), mode="RGB"
    )
    img3 = Image.fromarray(
        np.full((50, 50, 3), [0, 0, 255], dtype=np.uint8), mode="RGB"
    )
    return [img1, img2, img3]


@pytest.fixture
def class_mapping():
    """Standard class mapping for tests."""
    return {1: "cat", 2: "dog", 3: "bird"}


@pytest.fixture
def color_scheme():
    """Color scheme matching the class mapping."""
    return {"cat": (0, 0, 255), "dog": (0, 255, 0), "bird": (255, 0, 0)}


# -----------------------------------------------------------------------------
# Classification Predictor Tests
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Multi-Label Classification Predictor Tests
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Object Detection Predictor Tests
# -----------------------------------------------------------------------------


class TestBoundingBoxPredictor:
    """Tests for BoundingBoxPredictor and OnnxDetector."""

    @pytest.fixture
    def mock_detector_session(self, mock_onnx_session):
        """Creates a mock session configured for object detection."""
        return mock_onnx_session(
            resolution=640,
            input_names=["images", "target_sizes"],
            output_names=["boxes", "scores", "labels"],
        )

    def test_empty_input(self, mock_detector_session, class_mapping):
        """Predict with empty list returns empty list."""
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            result = predictor.predict([], confidence=0.5)

            assert result == []

    def test_preprocessing_target_sizes(
        self, mock_detector_session, sample_images, class_mapping
    ):
        """Target sizes contain original image dimensions (height, width)."""
        mock_detector_session.run_outputs = [
            np.zeros((3, 10, 4)),  # boxes
            np.zeros((3, 10)),  # scores
            np.zeros((3, 10), dtype=np.int64),  # labels
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            predictor.predict(sample_images, confidence=0.5)

            input_dict = mock_detector_session.run_inputs[0]
            target_sizes = input_dict["target_sizes"]

            # sample_images: (100, 150), (200, 100), (50, 50) - PIL size is (width, height)
            # target_sizes should be (height, width)
            assert target_sizes.shape == (3, 2)
            assert target_sizes[0].tolist() == [100, 150]  # height=100, width=150
            assert target_sizes[1].tolist() == [200, 100]  # height=200, width=100
            assert target_sizes[2].tolist() == [50, 50]  # height=50, width=50

    def test_preprocessing_image_resizing(
        self, mock_detector_session, sample_images, class_mapping
    ):
        """All images are resized to model resolution."""
        mock_detector_session.run_outputs = [
            np.zeros((3, 10, 4)),
            np.zeros((3, 10)),
            np.zeros((3, 10), dtype=np.int64),
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            predictor.predict(sample_images, confidence=0.5)

            input_dict = mock_detector_session.run_inputs[0]
            images_array = input_dict["images"]

            assert images_array.shape == (3, 640, 640, 3)

    def test_preprocessing_batch_ordering(self, mock_detector_session, class_mapping):
        """Images maintain their order in the batch."""
        # Create images with distinct colors
        red_img = Image.fromarray(
            np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8), mode="RGB"
        )
        green_img = Image.fromarray(
            np.full((100, 100, 3), [0, 255, 0], dtype=np.uint8), mode="RGB"
        )
        blue_img = Image.fromarray(
            np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8), mode="RGB"
        )

        mock_detector_session.run_outputs = [
            np.zeros((3, 10, 4)),
            np.zeros((3, 10)),
            np.zeros((3, 10), dtype=np.int64),
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            predictor.predict([red_img, green_img, blue_img], confidence=0.5)

            input_dict = mock_detector_session.run_inputs[0]
            images_array = input_dict["images"]

            # Check that dominant color is preserved in order
            assert images_array[0, 0, 0, 0] == 255  # red channel of first image
            assert images_array[1, 0, 0, 1] == 255  # green channel of second image
            assert images_array[2, 0, 0, 2] == 255  # blue channel of third image

    def test_postprocessing_confidence_filtering(
        self, mock_detector_session, class_mapping
    ):
        """Detections below confidence threshold are filtered out."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        # 3 detections: scores 0.9, 0.4, 0.6
        mock_detector_session.run_outputs = [
            np.array([[[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70]]]),
            np.array([[0.9, 0.4, 0.6]]),
            np.array([[1, 2, 1]]),
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            # Only detections with score > 0.5 should pass
            assert len(results[0]) == 2
            scores = [pred.score for pred in results[0]]
            assert 0.9 in scores
            assert 0.6 in scores
            assert 0.4 not in scores

    def test_postprocessing_class_mapping(self, mock_detector_session, class_mapping):
        """Class indices are correctly mapped to class names."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_detector_session.run_outputs = [
            np.array([[[10, 10, 50, 50], [20, 20, 60, 60]]]),
            np.array([[0.9, 0.8]]),
            np.array([[1, 3]]),  # cat=1, bird=3
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            classes = {pred.clss for pred in results[0]}
            assert classes == {"cat", "bird"}

    def test_postprocessing_unknown_class(self, mock_detector_session, class_mapping):
        """Unknown class indices map to 'Unknown'."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_detector_session.run_outputs = [
            np.array([[[10, 10, 50, 50]]]),
            np.array([[0.9]]),
            np.array([[99]]),  # Not in class mapping
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert results[0][0].clss == "Unknown"

    def test_postprocessing_nms(self, mock_detector_session, class_mapping):
        """NMS removes overlapping boxes."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        # Two highly overlapping boxes
        mock_detector_session.run_outputs = [
            np.array([[[10, 10, 50, 50], [12, 12, 52, 52]]]),
            np.array([[0.9, 0.8]]),
            np.array([[1, 1]]),
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)

            # Without NMS
            results_no_nms = predictor.predict(
                [img], confidence=0.5, nms_threshold=None
            )
            assert len(results_no_nms[0]) == 2

            # With NMS (low threshold should remove one)
            results_with_nms = predictor.predict(
                [img], confidence=0.5, nms_threshold=0.3
            )
            assert len(results_with_nms[0]) == 1
            assert results_with_nms[0][0].score == pytest.approx(0.9)

    def test_prediction_to_dict(self, mock_detector_session, class_mapping):
        """BoundingBoxPrediction.to_dict() produces expected structure."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_detector_session.run_outputs = [
            np.array([[[10.5, 20.5, 50.5, 60.5]]]),
            np.array([[0.85]]),
            np.array([[2]]),
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            result_dict = results[0][0].to_dict()
            assert result_dict["class"] == "dog"
            assert result_dict["score"] == pytest.approx(0.85)
            assert result_dict["bbox"]["x1"] == pytest.approx(10.5)
            assert result_dict["bbox"]["y1"] == pytest.approx(20.5)
            assert result_dict["bbox"]["x2"] == pytest.approx(50.5)
            assert result_dict["bbox"]["y2"] == pytest.approx(60.5)

    def test_annotation_bbox_drawing(
        self, mock_detector_session, checkerboard_image, class_mapping, color_scheme
    ):
        """Bounding box is drawn on the image at the correct location.

        Note: The color_scheme stores RGB colors but they are passed directly to
        cv2.rectangle which expects BGR. This means colors will be swapped.
        This test verifies current behavior, not ideal behavior.
        """
        # Box in top-left quadrant (0-50, 0-50 region)
        mock_detector_session.run_outputs = [
            np.array([[[5, 5, 45, 45]]]),
            np.array([[0.9]]),
            np.array([[1]]),  # cat
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_detector_session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", class_mapping)
            predictor.color_scheme = color_scheme

            results = predictor.predict([checkerboard_image], confidence=0.5)
            annotated = predictor.get_annotated_image(checkerboard_image, results[0])

            original_arr = np.array(checkerboard_image)
            annotated_arr = np.array(annotated)

            # Check a pixel on the top edge - verify it changed from original
            edge_pixel = annotated_arr[5, 25]  # y=5, x=25 (on top edge)
            original_edge = original_arr[5, 25]

            # The edge should have changed (a rectangle was drawn)
            assert not np.array_equal(
                edge_pixel, original_edge
            ), "Edge pixel should have changed"

            # Pixels well inside the box should be unchanged (white)
            interior_pixel = annotated_arr[25, 25]
            assert np.array_equal(interior_pixel, [255, 255, 255])

            # Pixels outside the bbox region should be unchanged
            outside_pixel = annotated_arr[75, 75]  # bottom-right quadrant
            assert np.array_equal(outside_pixel, [255, 255, 255])


# -----------------------------------------------------------------------------
# Instance Segmentation Predictor Tests
# -----------------------------------------------------------------------------


class TestInstanceSegmentationPredictor:
    """Tests for InstanceSegmentationPredictor and OnnxInstanceSegmentation."""

    @pytest.fixture
    def mock_segmentation_session(self, mock_onnx_session):
        """Creates a mock session configured for instance segmentation."""
        return mock_onnx_session(
            resolution=640,
            input_names=["images", "target_sizes"],
            output_names=["boxes", "scores", "labels", "masks"],
        )

    def test_empty_input(self, mock_segmentation_session, class_mapping):
        """Predict with empty list returns empty list."""
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_segmentation_session,
        ):
            predictor = InstanceSegmentationPredictor("fake.onnx", class_mapping)
            result = predictor.predict([], confidence=0.5)

            assert result == []

    def test_preprocessing_target_sizes(
        self, mock_segmentation_session, sample_images, class_mapping
    ):
        """Target sizes contain original image dimensions."""
        mock_segmentation_session.run_outputs = [
            np.zeros((3, 10, 4)),  # boxes
            np.zeros((3, 10)),  # scores
            np.zeros((3, 10), dtype=np.int64),  # labels
            np.zeros((3, 10, 28, 28)),  # masks (small resolution)
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_segmentation_session,
        ):
            predictor = InstanceSegmentationPredictor("fake.onnx", class_mapping)
            predictor.predict(sample_images, confidence=0.5)

            input_dict = mock_segmentation_session.run_inputs[0]
            target_sizes = input_dict["target_sizes"]

            assert target_sizes.shape == (3, 2)
            assert target_sizes[0].tolist() == [100, 150]
            assert target_sizes[1].tolist() == [200, 100]
            assert target_sizes[2].tolist() == [50, 50]

    def test_postprocessing_mask_resizing(
        self, mock_segmentation_session, class_mapping
    ):
        """Masks are resized from model output size to original image size."""
        img = Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8), mode="RGB")

        mock_segmentation_session.run_outputs = [
            np.array([[[10, 10, 290, 190]]]),
            np.array([[0.9]]),
            np.array([[1]]),
            np.ones((1, 1, 28, 28)) * 0.8,
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_segmentation_session,
        ):
            predictor = InstanceSegmentationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert results[0][0].mask.shape == (200, 300)

    def test_postprocessing_confidence_filtering(
        self, mock_segmentation_session, class_mapping
    ):
        """Detections below confidence threshold are filtered."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_segmentation_session.run_outputs = [
            np.array([[[10, 10, 50, 50], [20, 20, 60, 60], [30, 30, 70, 70]]]),
            np.array([[0.9, 0.3, 0.7]]),
            np.array([[1, 2, 1]]),
            np.ones((1, 3, 28, 28)) * 0.8,
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_segmentation_session,
        ):
            predictor = InstanceSegmentationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert len(results[0]) == 2
            scores = [pred.score for pred in results[0]]
            assert 0.9 in scores
            assert 0.7 in scores

    def test_postprocessing_empty_after_filtering(
        self, mock_segmentation_session, class_mapping
    ):
        """When all detections are filtered, returns empty list."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_segmentation_session.run_outputs = [
            np.array([[[10, 10, 50, 50]]]),
            np.array([[0.3]]),  # Below threshold
            np.array([[1]]),
            np.ones((1, 1, 28, 28)) * 0.8,
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_segmentation_session,
        ):
            predictor = InstanceSegmentationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert len(results[0]) == 0

    def test_prediction_to_dict(self, mock_segmentation_session, class_mapping):
        """InstanceSegmentationPrediction.to_dict() works with and without mask."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_segmentation_session.run_outputs = [
            np.array([[[10.5, 20.5, 50.5, 60.5]]]),
            np.array([[0.85]]),
            np.array([[2]]),
            np.ones((1, 1, 28, 28)) * 0.8,
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_segmentation_session,
        ):
            predictor = InstanceSegmentationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            result_dict = results[0][0].to_dict(include_mask=False)
            assert result_dict["class"] == "dog"
            assert result_dict["score"] == pytest.approx(0.85)
            assert "mask" not in result_dict

            result_dict_with_mask = results[0][0].to_dict(include_mask=True)
            assert "mask" in result_dict_with_mask

    def test_annotation_mask_overlay(
        self, mock_segmentation_session, checkerboard_image, class_mapping, color_scheme
    ):
        """Mask overlay blends colors in masked region.

        Note: Similar to bbox drawing, colors may be BGR/RGB swapped.
        This test verifies blending occurs, not specific color values.
        """
        mask = np.zeros((1, 1, 28, 28))
        mask[0, 0, :14, :14] = 1.0  # Top-left quadrant of mask

        mock_segmentation_session.run_outputs = [
            np.array([[[0, 0, 50, 50]]]),
            np.array([[0.9]]),
            np.array([[1]]),  # cat
            mask,
        ]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_segmentation_session,
        ):
            predictor = InstanceSegmentationPredictor("fake.onnx", class_mapping)
            predictor.color_scheme = color_scheme

            results = predictor.predict([checkerboard_image], confidence=0.5)
            annotated = predictor.get_annotated_image(checkerboard_image, results[0])

            original_arr = np.array(checkerboard_image)
            annotated_arr = np.array(annotated)

            # Check a pixel in the masked region (top-left quadrant)
            # Original was white (255, 255, 255), should be blended with mask color
            masked_pixel = annotated_arr[10, 10]
            original_pixel = original_arr[10, 10]

            # The masked pixel should be different from original (blending occurred)
            assert not np.array_equal(
                masked_pixel, original_pixel
            ), "Masked pixel should be blended"

            # Pixel outside mask region (bottom-right) - should be unchanged
            outside_pixel = annotated_arr[75, 75]
            assert np.array_equal(outside_pixel, original_arr[75, 75])


# -----------------------------------------------------------------------------
# Semantic Segmentation Predictor Tests
# -----------------------------------------------------------------------------


class TestSemanticSegmentationPredictor:
    """Tests for SemanticSegmentationPredictor and OnnxSemanticSegmentation."""

    @pytest.fixture
    def mock_semantic_session(self, mock_onnx_session):
        """Creates a mock session configured for semantic segmentation."""
        return mock_onnx_session(
            resolution=512,
            input_names=["images"],
            output_names=["masks"],
        )

    def test_empty_input(self, mock_semantic_session, class_mapping):
        """Predict with empty list returns empty list."""
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_semantic_session,
        ):
            predictor = SemanticSegmentationPredictor("fake.onnx", class_mapping)
            result = predictor.predict([])

            assert result == []

    def test_preprocessing_image_resizing(
        self, mock_semantic_session, sample_images, class_mapping
    ):
        """Images are resized to model resolution."""
        mock_semantic_session.run_outputs = [np.zeros((3, 3, 64, 64))]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_semantic_session,
        ):
            predictor = SemanticSegmentationPredictor("fake.onnx", class_mapping)
            predictor.predict(sample_images)

            input_dict = mock_semantic_session.run_inputs[0]
            images_array = input_dict["images"]

            assert images_array.shape == (3, 512, 512, 3)

    def test_postprocessing_mask_resizing(self, mock_semantic_session, class_mapping):
        """Output masks are resized to original image dimensions."""
        img = Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8), mode="RGB")

        mock_semantic_session.run_outputs = [np.zeros((1, 3, 64, 64))]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_semantic_session,
        ):
            predictor = SemanticSegmentationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img])

            assert results[0].class_mask.shape == (200, 300)

    def test_postprocessing_argmax_class_selection(
        self, mock_semantic_session, class_mapping
    ):
        """Class mask contains argmax of confidence masks."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        masks = np.zeros((1, 3, 50, 50))
        masks[0, 0, :25, :25] = 0.9  # Top-left: class 0
        masks[0, 1, :25, :25] = 0.1
        masks[0, 2, :25, :25] = 0.0
        masks[0, 0, 25:, 25:] = 0.1  # Bottom-right: class 2
        masks[0, 1, 25:, 25:] = 0.2
        masks[0, 2, 25:, 25:] = 0.8

        mock_semantic_session.run_outputs = [masks]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_semantic_session,
        ):
            predictor = SemanticSegmentationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img])

            class_mask = results[0].class_mask
            assert class_mask[10, 10] == 0  # Top-left should be class 0
            assert class_mask[90, 90] == 2  # Bottom-right should be class 2

    def test_prediction_to_dict(self, mock_semantic_session, class_mapping):
        """SemanticSegmentationPrediction.to_dict() works with and without conf_masks."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")

        mock_semantic_session.run_outputs = [np.zeros((1, 3, 50, 50))]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_semantic_session,
        ):
            predictor = SemanticSegmentationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img])

            result_dict = results[0].to_dict(include_conf_masks=False)
            assert "class_mask" in result_dict
            assert "conf_masks" not in result_dict

            result_dict_with_conf = results[0].to_dict(include_conf_masks=True)
            assert "conf_masks" in result_dict_with_conf

    def test_annotation_class_colors(
        self, mock_semantic_session, checkerboard_image, class_mapping, color_scheme
    ):
        """Annotation applies correct colors based on class mask."""
        masks = np.zeros((1, 3, 50, 50))
        masks[0, 1, :25, :] = 0.9  # Top half: class 1 (cat)
        masks[0, 2, :, :] = 0.5  # Bottom half: class 2 (dog)

        mock_semantic_session.run_outputs = [masks]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_semantic_session,
        ):
            predictor = SemanticSegmentationPredictor("fake.onnx", class_mapping)
            predictor.color_scheme = color_scheme

            results = predictor.predict([checkerboard_image])
            class_mask = results[0].class_mask
            annotated = predictor.get_annotated_image(checkerboard_image, class_mask)

            annotated_arr = np.array(annotated)

            # Top region blended toward cat color (red): 0.7*white + 0.3*red
            top_pixel = annotated_arr[10, 10]
            assert top_pixel[0] == 255
            assert 170 <= top_pixel[1] <= 185
            assert 170 <= top_pixel[2] <= 185

            # Bottom region blended toward dog color (green): 0.7*white + 0.3*green
            bottom_pixel = annotated_arr[75, 75]
            assert 170 <= bottom_pixel[0] <= 185
            assert bottom_pixel[1] == 255
            assert 170 <= bottom_pixel[2] <= 185


# -----------------------------------------------------------------------------
# VectorIndex Tests
# -----------------------------------------------------------------------------


class TestVectorIndex:
    """Tests for VectorIndex construction, search, aggregation, and serialization."""

    @pytest.fixture
    def normalized_vectors(self):
        np.random.seed(42)
        raw = np.random.randn(6, 64).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    @pytest.fixture
    def single_label_index(self, normalized_vectors):
        labels = [["A"], ["A"], ["A"], ["B"], ["B"], ["B"]]
        return VectorIndex(vectors=normalized_vectors, labels=labels)

    @pytest.fixture
    def multi_label_index(self, normalized_vectors):
        labels = [["A"], ["A"], ["A", "C"], ["B", "C"], ["B"], ["B"]]
        return VectorIndex(vectors=normalized_vectors, labels=labels)

    def test_construction_valid(self, normalized_vectors):
        labels = [["x"] for _ in range(6)]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        assert len(index) == 6
        assert index.dim == 64

    def test_construction_rejects_1d_vectors(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            VectorIndex(vectors=np.array([1.0, 2.0, 3.0]), labels=[["a"]])

    def test_construction_rejects_mismatched_lengths(self, normalized_vectors):
        with pytest.raises(ValueError, match="labels length"):
            VectorIndex(vectors=normalized_vectors, labels=[["a"], ["b"]])

    def test_construction_normalize(self):
        raw = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        index = VectorIndex(vectors=raw, labels=[["a"], ["b"]], normalize=True)
        norms = np.linalg.norm(index.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_construction_normalize_zero_vector(self):
        raw = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        index = VectorIndex(vectors=raw, labels=[["a"], ["b"]], normalize=True)
        np.testing.assert_array_equal(index.vectors[0], [0.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(index.vectors[1]), 1.0, atol=1e-6)

    def test_repr(self, multi_label_index):
        r = repr(multi_label_index)
        assert "6 vectors" in r
        assert "dim=64" in r
        assert "3 unique labels" in r

    # -- Search ---------------------------------------------------------------

    def test_search_self_is_top_match(self, single_label_index, normalized_vectors):
        results = single_label_index.search(normalized_vectors[0], k=1)
        assert len(results) == 1
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        assert results[0].labels == ["A"]

    def test_search_k_larger_than_index(self, single_label_index, normalized_vectors):
        results = single_label_index.search(normalized_vectors[0], k=100)
        assert len(results) == 6

    def test_search_returns_descending_scores(
        self, single_label_index, normalized_vectors
    ):
        results = single_label_index.search(normalized_vectors[0], k=6)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_multi_label_preserves_labels(
        self, multi_label_index, normalized_vectors
    ):
        results = multi_label_index.search(normalized_vectors[2], k=1)
        assert results[0].labels == ["A", "C"]

    def test_search_1d_and_2d_query_equivalent(
        self, single_label_index, normalized_vectors
    ):
        query = normalized_vectors[0]
        results_1d = single_label_index.search(query, k=3)
        results_2d = single_label_index.search(query.reshape(1, -1), k=3)
        for r1, r2 in zip(results_1d, results_2d):
            assert r1.score == pytest.approx(r2.score)
            assert r1.labels == r2.labels

    def test_search_batch(self, single_label_index, normalized_vectors):
        queries = normalized_vectors[:2]
        batch_results = single_label_index.search_batch(queries, k=2)
        assert len(batch_results) == 2
        assert len(batch_results[0]) == 2
        assert len(batch_results[1]) == 2
        assert batch_results[0][0].score == pytest.approx(1.0, abs=1e-5)
        assert batch_results[1][0].score == pytest.approx(1.0, abs=1e-5)

    # -- Aggregation ----------------------------------------------------------

    def test_aggregate_single_label(self, single_label_index):
        agg = single_label_index.aggregate()
        assert len(agg) == 2
        assert agg.labels == [["A"], ["B"]]
        norms = np.linalg.norm(agg.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_aggregate_multi_label(self, multi_label_index):
        agg = multi_label_index.aggregate()
        assert len(agg) == 3
        assert agg.labels == [["A"], ["B"], ["C"]]
        norms = np.linalg.norm(agg.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_aggregate_multi_label_centroid_correctness(
        self, multi_label_index, normalized_vectors
    ):
        """Label C appears on vectors 2 and 3. Its centroid should be the
        normalized mean of those two vectors."""
        agg = multi_label_index.aggregate()
        c_index = agg.labels.index(["C"])
        expected = normalized_vectors[2] + normalized_vectors[3]
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(agg.vectors[c_index], expected, atol=1e-5)

    def test_aggregate_already_unique(self, normalized_vectors):
        labels = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]
        index = VectorIndex(vectors=normalized_vectors, labels=labels)
        agg = index.aggregate()
        assert len(agg) == 6
        np.testing.assert_allclose(agg.vectors, normalized_vectors, atol=1e-6)

    # -- Dump / Load ----------------------------------------------------------

    def test_dump_and_load_roundtrip(self, single_label_index, tmp_path):
        single_label_index.dump(str(tmp_path))
        loaded = load_vector_index(str(tmp_path))
        assert len(loaded) == len(single_label_index)
        assert loaded.labels == single_label_index.labels
        np.testing.assert_allclose(loaded.vectors, single_label_index.vectors)

    def test_dump_and_load_multi_label_roundtrip(self, multi_label_index, tmp_path):
        multi_label_index.dump(str(tmp_path))
        loaded = load_vector_index(str(tmp_path))
        assert loaded.labels == multi_label_index.labels
        np.testing.assert_allclose(loaded.vectors, multi_label_index.vectors)

    def test_dump_creates_expected_files(self, single_label_index, tmp_path):
        files = single_label_index.dump(str(tmp_path))
        assert len(files) == 2
        assert all(os.path.exists(f) for f in files)
        assert any(f.endswith(".yaml") for f in files)
        assert any(f.endswith(".npz") for f in files)


# -----------------------------------------------------------------------------
# build_vector_index Tests
# -----------------------------------------------------------------------------


class TestBuildVectorIndex:
    """Tests for the build_vector_index factory function."""

    @pytest.fixture
    def mock_feature_extractor(self):
        extractor = MagicMock()

        def fake_predict(images):
            results = []
            for _ in images:
                mock_result = MagicMock()
                mock_result.feature = np.random.randn(64).astype(np.float32)
                results.append(mock_result)
            return results

        extractor.predict.side_effect = fake_predict
        return extractor

    def test_build_single_label(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(4)
        ]
        labels = ["A", "A", "B", "B"]
        index = build_vector_index(crops, labels, mock_feature_extractor)
        assert len(index) == 4
        assert index.labels == [["A"], ["A"], ["B"], ["B"]]

    def test_build_multi_label(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(3)
        ]
        labels = [["A", "B"], ["B"], ["C"]]
        index = build_vector_index(
            crops, labels, mock_feature_extractor, multi_label=True
        )
        assert len(index) == 3
        assert index.labels == [["A", "B"], ["B"], ["C"]]

    def test_build_from_file_paths(self, mock_feature_extractor, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"crop_{i}.png"
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(str(p))
            paths.append(str(p))
        labels = ["A", "B", "C"]
        index = build_vector_index(paths, labels, mock_feature_extractor)
        assert len(index) == 3

    def test_build_rejects_mismatched_lengths(self, mock_feature_extractor):
        crops = [Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))]
        with pytest.raises(ValueError, match="crops length"):
            build_vector_index(crops, ["A", "B"], mock_feature_extractor)

    def test_build_rejects_bad_crop_type(self, mock_feature_extractor):
        with pytest.raises(
            TypeError, match="PIL Image, a file path string, or a CropSpec"
        ):
            build_vector_index([12345], ["A"], mock_feature_extractor)

    def test_build_batching(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(5)
        ]
        labels = ["A"] * 5
        build_vector_index(crops, labels, mock_feature_extractor, batch_size=2)
        assert mock_feature_extractor.predict.call_count == 3  # 2 + 2 + 1

    def test_build_normalizes_by_default(self, mock_feature_extractor):
        crops = [
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)) for _ in range(3)
        ]
        labels = ["A", "B", "C"]
        index = build_vector_index(crops, labels, mock_feature_extractor)
        norms = np.linalg.norm(index.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_build_from_crop_specs(self, mock_feature_extractor, tmp_path):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        p = tmp_path / "full_image.png"
        img.save(str(p))
        specs = [
            CropSpec(path=str(p), bbox=(0, 0, 50, 50)),
            CropSpec(path=str(p), bbox=(50, 50, 100, 100)),
        ]
        index = build_vector_index(specs, ["A", "B"], mock_feature_extractor)
        assert len(index) == 2

    def test_build_mixed_crop_types(self, mock_feature_extractor, tmp_path):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        p = tmp_path / "image.png"
        img.save(str(p))
        crops = [
            img,
            str(p),
            CropSpec(path=str(p), bbox=(10, 10, 50, 50)),
        ]
        index = build_vector_index(crops, ["A", "B", "C"], mock_feature_extractor)
        assert len(index) == 3

    def test_crop_spec_crops_correctly(self, tmp_path):
        """Verify CropSpec actually crops to the specified bbox."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[20:40, 30:60] = [255, 0, 0]
        img = Image.fromarray(arr)
        p = tmp_path / "image.png"
        img.save(str(p))

        from orient_express.predictors.vector_index import _CropDataset

        dataset = _CropDataset([CropSpec(path=str(p), bbox=(30, 20, 60, 40))])
        crop = dataset[0]
        crop_arr = np.array(crop)
        assert crop.size == (30, 20)
        assert np.all(crop_arr[:, :, 0] == 255)
