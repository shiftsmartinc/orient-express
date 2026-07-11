"""Instance segmentation predictor tests."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors.instance_segmentation import (
    InstanceSegmentationPredictor,
)


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
            assert not np.array_equal(masked_pixel, original_pixel), (
                "Masked pixel should be blended"
            )

            # Pixel outside mask region (bottom-right) - should be unchanged
            outside_pixel = annotated_arr[75, 75]
            assert np.array_equal(outside_pixel, original_arr[75, 75])
