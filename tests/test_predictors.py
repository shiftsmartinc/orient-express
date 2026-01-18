"""
Tests for predictor classes.

These tests verify that:
1. Preprocessing produces correctly shaped and ordered arrays
2. Target sizes contain original image dimensions
3. Postprocessing correctly filters, formats, and resizes outputs
4. Dataclass formatting produces expected structures
5. Annotation methods modify images correctly in expected regions
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

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
        mock_classifier_session.run_outputs = [np.array(
            [[0.1, 0.7, 0.2], [0.1, 0.7, 0.2], [0.1, 0.7, 0.2]]
        )]

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
            np.array([
                [0.9, 0.1, 0.1],  # Image 1: only cat
                [0.1, 0.9, 0.9],  # Image 2: dog and bird
            ])
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
