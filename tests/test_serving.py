"""Tests for orient_express.serving helpers and predictor to_response shapes."""

import json

import pytest

from orient_express.serving import build_predict_kwargs, decode_input


class TestDecodeInput:
    def test_decodes_json_string(self):
        assert decode_input('{"instances": []}') == {"instances": []}

    def test_decodes_json_bytes(self):
        assert decode_input(b'{"instances": []}') == {"instances": []}

    def test_passes_dict_through(self):
        payload = {"instances": [1]}
        assert decode_input(payload) is payload

    def test_rejects_other_types(self):
        with pytest.raises(Exception, match="unsupported payload type"):
            decode_input(42)


class TestBuildPredictKwargs:
    def test_required_confidence_gets_server_default(self):
        def predict(images, confidence):
            pass

        assert build_predict_kwargs(predict, {}) == {"confidence": 0.5}

    def test_request_parameter_overrides_default(self):
        def predict(images, confidence):
            pass

        assert build_predict_kwargs(predict, {"confidence": 0.9}) == {"confidence": 0.9}

    def test_optional_parameter_forwarded_when_present(self):
        def predict(images, confidence, nms_threshold=None):
            pass

        kwargs = build_predict_kwargs(
            predict, {"confidence": 0.4, "nms_threshold": 0.3}
        )
        assert kwargs == {"confidence": 0.4, "nms_threshold": 0.3}

    def test_optional_parameter_omitted_when_absent(self):
        def predict(images, confidence, nms_threshold=None):
            pass

        assert build_predict_kwargs(predict, {}) == {"confidence": 0.5}

    def test_no_parameters_for_bare_predict(self):
        def predict(images):
            pass

        assert build_predict_kwargs(predict, {"confidence": 0.9}) == {}

    def test_reserved_and_unknown_keys_ignored(self):
        def predict(images, confidence):
            pass

        kwargs = build_predict_kwargs(
            predict, {"confidence": 0.7, "debug_image": False, "bogus": 1}
        )
        assert kwargs == {"confidence": 0.7}

    def test_defaulted_confidence_still_filled_from_request(self):
        def predict(images, confidence=0.5):
            pass

        assert build_predict_kwargs(predict, {"confidence": 0.8}) == {"confidence": 0.8}


class TestToResponse:
    def _fake_detector(self):
        """A BoundingBoxPredictor with the ONNX session swapped for a stub."""
        from unittest.mock import MagicMock, patch

        import numpy as np

        from orient_express.predictors import BoundingBoxPredictor

        session = MagicMock()
        session.get_inputs.return_value = [
            MagicMock(name="images", shape=[1, 640, 640, 3])
        ]
        session.get_outputs.return_value = []
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=session,
        ):
            predictor = BoundingBoxPredictor("fake.onnx", {1: "cat"})
        return predictor, np.array([[0, 0, 10, 10, 0.9, 1]])

    def test_detection_response_shape_with_debug(self):
        from PIL import Image

        predictor, boxes = self._fake_detector()
        image = Image.new("RGB", (32, 32))
        prediction = predictor.format_output(boxes)

        response = predictor.to_response(image, prediction, include_debug=True)
        assert response["status"] == "success"
        assert isinstance(response["predictions"], list)
        assert response["predictions"][0]["class"] == "cat"
        assert isinstance(response["debug_image"], str)
        json.dumps(response)  # response must be JSON-serializable

    def test_detection_response_skips_debug_when_disabled(self):
        from PIL import Image

        predictor, boxes = self._fake_detector()
        image = Image.new("RGB", (32, 32))
        prediction = predictor.format_output(boxes)

        response = predictor.to_response(image, prediction, include_debug=False)
        assert "debug_image" not in response
        assert response["status"] == "success"

    def test_classification_response_is_flat(self):
        from unittest.mock import MagicMock, patch

        from PIL import Image

        from orient_express.predictors import (
            ClassificationPrediction,
            ClassificationPredictor,
        )

        session = MagicMock()
        session.get_inputs.return_value = [
            MagicMock(name="images", shape=[1, 224, 224, 3])
        ]
        session.get_outputs.return_value = []
        with patch(
            "orient_express.predictors.predictor.ort.InferenceSession",
            return_value=session,
        ):
            predictor = ClassificationPredictor("fake.onnx", {1: "cat"})

        prediction = ClassificationPrediction(
            clss="cat", score=0.9, class_scores={"cat": 0.9}
        )
        response = predictor.to_response(Image.new("RGB", (8, 8)), prediction)
        assert response == {
            "class": "cat",
            "score": 0.9,
            "class_scores": {"cat": 0.9},
            "status": "success",
        }
