"""Bounding-box (object detection) predictor tests."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors.object_detection import (
    BoundingBoxPredictor,
)


class TestBoundingBoxPredictor:
    """Tests for BoundingBoxPredictor."""

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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            "orient_express.predictors.runtime.ort.InferenceSession",
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
            assert not np.array_equal(edge_pixel, original_edge), (
                "Edge pixel should have changed"
            )

            # Pixels well inside the box should be unchanged (white)
            interior_pixel = annotated_arr[25, 25]
            assert np.array_equal(interior_pixel, [255, 255, 255])

            # Pixels outside the bbox region should be unchanged
            outside_pixel = annotated_arr[75, 75]  # bottom-right quadrant
            assert np.array_equal(outside_pixel, [255, 255, 255])


class TestNmsHelper:
    """Pins the behavior of the cv2-backed nms() (verified against torchvision.ops.nms during the torch removal: 300/300 random keep-set matches; see that PR for the comparison harness)."""

    def test_orders_by_descending_score(self):
        from orient_express.predictors.object_detection import nms

        boxes = np.array(
            [[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]],
            dtype=np.float32,
        )
        scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
        assert list(nms(boxes, scores, 0.5)) == [1, 2, 0]

    def test_suppresses_overlapping_lower_score(self):
        from orient_express.predictors.object_detection import nms

        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        assert list(nms(boxes, scores, 0.5)) == [0, 2]

    def test_high_threshold_keeps_all(self):
        from orient_express.predictors.object_detection import nms

        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        assert len(nms(boxes, scores, 0.99)) == 2

    def test_empty_input(self):
        from orient_express.predictors.object_detection import nms

        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        assert len(nms(boxes, scores, 0.5)) == 0
