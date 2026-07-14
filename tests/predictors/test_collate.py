"""Tests for OnnxSessionWrapper.collate_images (preallocated + threaded)."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image

from orient_express.predictors import predictor as predictor_module
from orient_express.predictors.predictor import OnnxSessionWrapper
from orient_express.utils.image_processor import image_to_array


def make_wrapper(resolution=64):
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(shape=[1, resolution, resolution, 3])]
    session.get_outputs.return_value = []
    with patch(
        "orient_express.predictors.predictor.ort.InferenceSession",
        return_value=session,
    ):
        return OnnxSessionWrapper("fake.onnx")


def reference_collate(wrapper, pil_images):
    """The original list-based implementation."""
    images = [cv2.resize(image_to_array(img), wrapper.img_size) for img in pil_images]
    return np.array(images)


def random_images(n, size=(100, 80)):
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        for _ in range(n)
    ]


def test_serial_path_matches_reference():
    wrapper = make_wrapper()
    images = random_images(3)  # small: below threading threshold
    np.testing.assert_array_equal(
        wrapper.collate_images(images), reference_collate(wrapper, images)
    )


def test_threaded_path_matches_reference(monkeypatch):
    wrapper = make_wrapper()
    images = random_images(8)
    # force the threaded path regardless of image sizes
    monkeypatch.setattr(predictor_module, "THREADED_COLLATE_MIN_TOTAL_PIXELS", 1)
    np.testing.assert_array_equal(
        wrapper.collate_images(images), reference_collate(wrapper, images)
    )


def test_output_shape_and_dtype():
    wrapper = make_wrapper(resolution=32)
    batch = wrapper.collate_images(random_images(5))
    assert batch.shape == (5, 32, 32, 3)
    assert batch.dtype == np.uint8


def test_empty_batch():
    wrapper = make_wrapper()
    assert wrapper.collate_images([]).shape == (0, 64, 64, 3)
