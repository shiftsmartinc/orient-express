"""Semantic segmentation predictor tests."""

import base64
import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors.semantic_segmentation import (
    SemanticSegmentationPredictor,
)


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
            valid_mask = results[0].valid_mask
            assert class_mask.dtype == np.uint8
            assert valid_mask.dtype == bool
            assert class_mask[10, 10] == 0  # Top-left should be class 0
            assert class_mask[90, 90] == 2  # Bottom-right should be class 2
            assert valid_mask[10, 10]  # max prob 0.9 >= 0.5
            assert valid_mask[90, 90]  # max prob 0.8 >= 0.5
            assert not valid_mask[50, 50]  # zeros region, below threshold

    def test_confidence_threshold(self, mock_semantic_session, class_mapping):
        """valid_mask reflects the confidence threshold."""
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")
        masks = np.full((1, 3, 50, 50), 0.1)  # all classes below 0.5

        mock_semantic_session.run_outputs = [masks]

        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=mock_semantic_session,
        ):
            predictor = SemanticSegmentationPredictor("fake.onnx", class_mapping)
            results = predictor.predict([img], confidence=0.5)

            assert not results[0].valid_mask.any()

            # Lower threshold makes every pixel valid.
            results = predictor.predict([img], confidence=0.05)
            assert results[0].valid_mask.all()

    def test_prediction_to_dict(self, mock_semantic_session, class_mapping):
        """SemanticSegmentationPrediction.to_dict() emits base64 PNGs for masks."""
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
            assert "valid_mask" in result_dict
            assert "conf_masks" not in result_dict

            class_png = Image.open(
                io.BytesIO(base64.b64decode(result_dict["class_mask"]))
            )
            valid_png = Image.open(
                io.BytesIO(base64.b64decode(result_dict["valid_mask"]))
            )
            assert class_png.size == (100, 100)
            assert valid_png.size == (100, 100)

            result_dict_with_conf = results[0].to_dict(include_conf_masks=True)
            assert "conf_masks" in result_dict_with_conf
            decoded_conf = np.load(
                io.BytesIO(base64.b64decode(result_dict_with_conf["conf_masks"]))
            )
            np.testing.assert_array_equal(decoded_conf, results[0].conf_masks)

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
            annotated = predictor.get_annotated_image(checkerboard_image, results[0])

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
