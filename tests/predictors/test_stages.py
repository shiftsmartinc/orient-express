"""Tests for the staged inference API (preprocess / infer / postprocess)."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors.object_detection import BoundingBoxPredictor

RESOLUTION = 64


def make_images(n, size=(80, 60)):
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        for _ in range(n)
    ]


def detection_outputs(batch_size, score=0.9):
    return [
        np.tile([10.0, 10.0, 50.0, 50.0], (batch_size, 1, 1)),  # boxes
        np.full((batch_size, 1), score),  # scores
        np.ones((batch_size, 1), dtype=np.int64),  # labels
    ]


@pytest.fixture
def detector(mock_onnx_session, class_mapping):
    session = mock_onnx_session(
        resolution=RESOLUTION,
        input_names=["images", "target_sizes"],
        output_names=["boxes", "scores", "labels"],
    )

    def run(output_names, input_dict):
        session.run_inputs.append(input_dict)
        return detection_outputs(len(input_dict["images"]))

    session.run.side_effect = run
    with patch(
        "orient_express.predictors.runtime.ort.InferenceSession",
        return_value=session,
    ):
        yield BoundingBoxPredictor("fake.onnx", class_mapping)


def test_stages_compose_to_predict(detector):
    images = make_images(2)
    feed = detector.preprocess(images)
    assert set(feed) == {"images", "target_sizes"}
    assert feed["images"].shape == (2, RESOLUTION, RESOLUTION, 3)

    outputs = detector.infer(feed)
    staged = detector.postprocess(outputs, feed, confidence=0.5)
    allinone = detector.predict(images, confidence=0.5)

    assert len(staged) == len(allinone) == 2
    for a, b in zip(staged, allinone, strict=True):
        assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


def test_infer_only_feeds_model_inputs(detector):
    images = make_images(1)
    feed = detector.preprocess(images)
    feed["extra_context"] = np.zeros(1)
    detector.infer(feed)
    sent = detector.session.run_inputs[-1]
    assert set(sent) == {"images", "target_sizes"}


def test_device_constants_cover_supported_devices():
    from orient_express.predictors import Device
    from orient_express.predictors.runtime import _DEVICE_TO_PROVIDER

    constants = {v for k, v in vars(Device).items() if not k.startswith("_")}
    assert constants == set(_DEVICE_TO_PROVIDER)


def test_unknown_device_rejected(mock_onnx_session, class_mapping):
    session = mock_onnx_session(
        resolution=RESOLUTION, input_names=["images"], output_names=["boxes"]
    )
    with patch(
        "orient_express.predictors.runtime.ort.InferenceSession",
        return_value=session,
    ):
        with pytest.raises(ValueError, match="Unknown device 'mps'"):
            BoundingBoxPredictor("fake.onnx", class_mapping, device="mps")


def test_gpu_fallback_fails_loudly(mock_onnx_session, class_mapping):
    # ORT silently serves on CPU when a GPU provider can't load; that is a
    # 10-50x slowdown that looks like a working deployment
    session = mock_onnx_session(
        resolution=RESOLUTION, input_names=["images"], output_names=["boxes"]
    )
    session.get_providers.return_value = ["CPUExecutionProvider"]
    with (
        patch(
            "orient_express.predictors.runtime.ort.InferenceSession",
            return_value=session,
        ),
        patch("orient_express.predictors.runtime._preload_gpu_dlls"),
    ):
        with pytest.raises(RuntimeError, match="activated CPUExecutionProvider"):
            BoundingBoxPredictor("fake.onnx", class_mapping, device="cuda")


def test_nchw_model_rejected(mock_onnx_session, class_mapping):
    # the library contract is NHWC; an NCHW export would silently become
    # resolution=3 and produce garbage resizes
    session = mock_onnx_session(
        resolution=RESOLUTION, input_names=["images"], output_names=["boxes"]
    )
    session.get_inputs.return_value[0].shape = [None, 3, 640, 640]
    with patch(
        "orient_express.predictors.runtime.ort.InferenceSession",
        return_value=session,
    ):
        with pytest.raises(ValueError, match="NHWC"):
            BoundingBoxPredictor("fake.onnx", class_mapping)


def test_get_predictor_forwards_kwargs(tmp_path, class_mapping):
    import yaml

    from orient_express.predictors import get_predictor
    from orient_express.utils.paths import get_metadata_path

    metadata = {
        "model_type": "object-detection-onnx",
        "model_file": "model.onnx",
        "classes": class_mapping,
    }
    with open(get_metadata_path(str(tmp_path)), "w") as f:
        yaml.dump(metadata, f)

    with patch.object(BoundingBoxPredictor, "from_dir") as from_dir:
        get_predictor(
            str(tmp_path),
            "cpu",
            provider_options={"cudnn_conv_algo_search": "EXHAUSTIVE"},
        )
    from_dir.assert_called_once_with(
        str(tmp_path),
        metadata,
        "cpu",
        provider_options={"cudnn_conv_algo_search": "EXHAUSTIVE"},
    )


def test_get_predictor_joblib_rejects_kwargs(tmp_path):
    import yaml

    from orient_express.predictors import get_predictor
    from orient_express.utils.paths import get_metadata_path

    with open(get_metadata_path(str(tmp_path)), "w") as f:
        yaml.dump({"model_type": "joblib", "model_file": "m.joblib"}, f)
    with pytest.raises(TypeError, match="joblib"):
        get_predictor(str(tmp_path), provider_options={})


def test_infer_schedules_cache_upload_only_on_new_shapes(detector):
    from unittest.mock import MagicMock

    detector._trt_cache_sync = MagicMock()
    detector.predict(make_images(2), confidence=0.5)
    detector.predict(make_images(2), confidence=0.5)  # same shapes: no build
    assert detector._trt_cache_sync.schedule_upload.call_count == 1
    detector.predict(make_images(3), confidence=0.5)  # new batch size
    assert detector._trt_cache_sync.schedule_upload.call_count == 2


def test_parse_trt_profile_shapes():
    from orient_express.predictors.runtime import parse_trt_profile_shapes

    assert parse_trt_profile_shapes("images:1x576x576x3,target_sizes:1x2") == {
        "images": [1, 576, 576, 3],
        "target_sizes": [1, 2],
    }
    with pytest.raises(ValueError, match="Malformed"):
        parse_trt_profile_shapes("garbage")


def test_tensorrt_requires_profile(class_mapping, tmp_path, monkeypatch):
    # profiles are mandatory for TRT devices: without one, TRT compiles for
    # the first shape it sees and any other shape means a silent rebuild.
    # Both checks fire before any session exists.
    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ORIENT_EXPRESS_TRT_CACHE_GCS", raising=False)
    model_path = tmp_path / "fake.onnx"
    model_path.write_bytes(b"weights")
    with patch("orient_express.predictors.runtime.ort.InferenceSession") as session_cls:
        with pytest.raises(ValueError, match="requires an explicit TensorRT"):
            BoundingBoxPredictor(str(model_path), class_mapping, device="tensorrt")
        with pytest.raises(ValueError, match="missing: trt_profile_opt_shapes"):
            BoundingBoxPredictor(
                str(model_path),
                class_mapping,
                device="tensorrt",
                provider_options={
                    "trt_profile_min_shapes": "images:1x64x64x3,target_sizes:1x2"
                },
            )
        session_cls.assert_not_called()


def test_malformed_profile_rejected_before_session_build(class_mapping):
    # ORT ignores a malformed spec with only a log warning (the engine then
    # profiles the first shape it sees instead of the intended range), and a
    # late parse error would come after the multi-minute engine build — so
    # the spec must be validated before any session is created.
    with patch("orient_express.predictors.runtime.ort.InferenceSession") as session_cls:
        with pytest.raises(ValueError, match="Malformed"):
            BoundingBoxPredictor(
                "fake.onnx",
                class_mapping,
                device="tensorrt",
                provider_options={"trt_profile_min_shapes": "garbage"},
            )
        session_cls.assert_not_called()


def test_trt_explicit_profile_enforced(
    mock_onnx_session, class_mapping, tmp_path, monkeypatch
):
    monkeypatch.setenv("ORIENT_EXPRESS_TRT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ORIENT_EXPRESS_TRT_CACHE_GCS", raising=False)
    session = mock_onnx_session(
        resolution=RESOLUTION,
        input_names=["images", "target_sizes"],
        output_names=["boxes", "scores", "labels"],
    )
    session.get_providers.return_value = ["TensorrtExecutionProvider"]

    def run(output_names, input_dict):
        return detection_outputs(len(input_dict["images"]))

    session.run.side_effect = run
    model_path = tmp_path / "fake.onnx"
    model_path.write_bytes(b"weights")  # trt_cache_scope hashes the file
    with (
        patch(
            "orient_express.predictors.runtime.ort.InferenceSession",
            return_value=session,
        ),
        patch("orient_express.predictors.runtime._preload_gpu_dlls"),
        patch("orient_express.predictors.runtime._preload_tensorrt_libs"),
    ):
        predictor = BoundingBoxPredictor(
            str(model_path),
            class_mapping,
            device="tensorrt",
            provider_options={
                "trt_profile_min_shapes": "images:1x64x64x3,target_sizes:1x2",
                "trt_profile_opt_shapes": "images:8x64x64x3,target_sizes:8x2",
                "trt_profile_max_shapes": "images:8x64x64x3,target_sizes:8x2",
            },
        )

    predictor.predict(make_images(8), confidence=0.5)  # at the max: fine
    with pytest.raises(ValueError, match="outside the declared optimization"):
        predictor.predict(make_images(9), confidence=0.5)
