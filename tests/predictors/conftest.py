"""Shared fixtures for predictor tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def mock_onnx_session():
    """Creates a mock ONNX session factory that captures inputs and returns configured outputs.

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
    """Creates a 100x100 checkerboard image with 50x50 quadrants.

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
