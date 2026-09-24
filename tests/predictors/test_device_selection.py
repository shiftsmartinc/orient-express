"""Serving-side device support: trt_batch profiles and batch bounds.

The device itself is chosen at deploy time (see vertex.deploy_to_endpoint);
these cover the two things a caller handed only a device string needs.
"""

from unittest.mock import MagicMock, patch

import pytest

from orient_express.predictors import Device, runtime
from orient_express.predictors import predictor as predictor_module
from orient_express.predictors.predictor import ImagePredictor
from orient_express.predictors.runtime import (
    _normalize_trt_batch,
    apply_trt_batch,
    trt_profile_for_batch,
)

# --------------------------------------------------------- batch normalizing


def test_bare_int_pins_opt_to_the_dominant_shape():
    # min stays 1 so a ragged final batch stays in profile
    assert _normalize_trt_batch(8) == (1, 8, 8)


def test_triple_passes_through():
    assert _normalize_trt_batch((1, 1, 8)) == (1, 1, 8)


@pytest.mark.parametrize("bad", [0, -1, (0, 1, 8), (1, 9, 8), (4, 1, 8)])
def test_rejects_impossible_ranges(bad):
    with pytest.raises(ValueError, match="trt_batch"):
        _normalize_trt_batch(bad)


@pytest.mark.parametrize("bad", ["8", (1, 8), None])
def test_rejects_malformed(bad):
    with pytest.raises(ValueError, match="trt_batch"):
        _normalize_trt_batch(bad)


# ------------------------------------------------------------ profile build


def _session(shapes: dict):
    session = MagicMock()
    inputs = []
    for name, shape in shapes.items():
        inp = MagicMock()
        inp.name = name
        inp.shape = shape
        inputs.append(inp)
    session.get_inputs.return_value = inputs
    return session


def _profile(shapes: dict, trt_batch):
    with patch.object(runtime.ort, "InferenceSession", return_value=_session(shapes)):
        return trt_profile_for_batch("model.onnx", trt_batch)


def test_batch_fans_out_to_every_input_sharing_the_symbolic_dim():
    # detection: images and target_sizes are the SAME onnx symbolic dim, so
    # one bound has to reach both
    profile = _profile(
        {"images": ["batch_size", 576, 576, 3], "target_sizes": ["batch_size", 2]},
        (1, 1, 8),
    )
    assert profile == {
        "trt_profile_min_shapes": "images:1x576x576x3,target_sizes:1x2",
        "trt_profile_opt_shapes": "images:1x576x576x3,target_sizes:1x2",
        "trt_profile_max_shapes": "images:8x576x576x3,target_sizes:8x2",
    }


def test_single_input_model():
    profile = _profile({"image": ["batch", 256, 256, 3]}, 4)
    assert profile["trt_profile_min_shapes"] == "image:1x256x256x3"
    assert profile["trt_profile_opt_shapes"] == "image:4x256x256x3"
    assert profile["trt_profile_max_shapes"] == "image:4x256x256x3"


def test_symbolic_dim_is_matched_by_name_not_position():
    profile = _profile({"tokens": [3, "batch"]}, (1, 1, 8))
    assert profile["trt_profile_max_shapes"] == "tokens:3x8"


def test_two_distinct_symbolic_dims_are_ambiguous():
    with pytest.raises(ValueError, match="more than one dynamic dimension"):
        _profile({"images": ["batch", 576, 576, 3], "q": ["queries", 4]}, 8)


def test_static_export_rejects_a_batch_it_cannot_serve():
    with pytest.raises(ValueError, match="static export pinned at batch 1"):
        _profile({"images": [1, 432, 432, 3], "target_sizes": [1, 2]}, 8)


def test_static_export_accepts_its_own_pinned_batch():
    profile = _profile({"images": [1, 432, 432, 3], "target_sizes": [1, 2]}, 1)
    assert profile["trt_profile_max_shapes"] == "images:1x432x432x3,target_sizes:1x2"


# ------------------------------------------------------------- apply_trt_batch


def test_trt_batch_is_meaningless_off_tensorrt():
    with pytest.raises(ValueError, match="does not apply to device='cuda'"):
        apply_trt_batch("model.onnx", Device.CUDA, 8, None)


def test_trt_batch_and_explicit_shapes_conflict():
    with pytest.raises(ValueError, match="not both"):
        apply_trt_batch(
            "model.onnx",
            Device.TENSORRT,
            8,
            {"trt_profile_min_shapes": "images:1x64x64x3"},
        )


def test_unrelated_provider_options_survive():
    with patch.object(
        runtime.ort,
        "InferenceSession",
        return_value=_session({"image": ["batch", 64, 64, 3]}),
    ):
        options = apply_trt_batch(
            "model.onnx",
            Device.TENSORRT,
            2,
            {"trt_builder_optimization_level": 5},
        )
    assert options["trt_builder_optimization_level"] == 5
    assert options["trt_profile_max_shapes"] == "image:2x64x64x3"


def test_predictor_accepts_trt_batch_end_to_end():
    """trt_batch reaches the session as profile options.

    The resulting bounds also drive max_batch_size.
    """
    session = _session({"image": ["batch", 64, 64, 3]})
    session.get_outputs.return_value = []
    with (
        patch.object(runtime.ort, "InferenceSession", return_value=session),
        patch.object(
            predictor_module, "create_session", return_value=(session, None)
        ) as create,
    ):
        predictor = ImagePredictor(
            "model.onnx", device=Device.TENSORRT_FP16, trt_batch=(1, 1, 8)
        )
    options = create.call_args.args[2]
    assert options["trt_profile_max_shapes"] == "image:8x64x64x3"
    assert predictor.max_batch_size == 8


# --------------------------------------------------------- max_batch_size


def _image_predictor(device, provider_options=None):
    session = _session({"images": [None, 64, 64, 3]})
    session.get_outputs.return_value = []
    with patch.object(predictor_module, "create_session", return_value=(session, None)):
        return ImagePredictor(
            "model.onnx", device=device, provider_options=provider_options
        )


def test_max_batch_size_from_trt_profile():
    predictor = _image_predictor(
        Device.TENSORRT,
        {
            "trt_profile_min_shapes": "images:1x64x64x3,target_sizes:1x2",
            "trt_profile_opt_shapes": "images:1x64x64x3,target_sizes:1x2",
            "trt_profile_max_shapes": "images:8x64x64x3,target_sizes:8x2",
        },
    )
    assert predictor.max_batch_size == 8


def test_max_batch_size_none_without_profile():
    assert _image_predictor(Device.CPU).max_batch_size is None


# --------------------------------------------------- graph_optimizations


def _capture_session_options(**predictor_kwargs):
    session = _session({"images": [None, 64, 64, 3]})
    session.get_outputs.return_value = []
    with (
        patch.object(runtime.ort, "InferenceSession", return_value=session) as make,
        patch.object(runtime, "_preload_gpu_dlls"),
    ):
        session.get_providers.return_value = ["CPUExecutionProvider"]
        ImagePredictor("model.onnx", **predictor_kwargs)
    return make.call_args.kwargs["sess_options"]


def test_graph_optimizations_on_by_default():
    options = _capture_session_options(device=Device.CPU)
    assert options.graph_optimization_level == (
        runtime.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )


def test_graph_optimizations_can_be_disabled():
    """The serving container disables fusion: ORT 1.24.x mis-fuses DINOv3."""
    options = _capture_session_options(device=Device.CPU, graph_optimizations=False)
    assert options.graph_optimization_level == (
        runtime.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
