"""Tests for the staged inference API and predict_stream pipelining."""

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


def test_predict_stream_matches_predict_in_order(detector):
    batches = [make_images(1), make_images(3), make_images(2)]
    streamed = list(detector.predict_stream(batches, confidence=0.5))
    direct = [detector.predict(b, confidence=0.5) for b in batches]

    assert [len(r) for r in streamed] == [1, 3, 2]
    for s_batch, d_batch in zip(streamed, direct, strict=True):
        for s, d in zip(s_batch, d_batch, strict=True):
            assert [p.to_dict() for p in s] == [p.to_dict() for p in d]


def test_predict_stream_empty_iterable(detector):
    assert list(detector.predict_stream([], confidence=0.5)) == []


def test_predict_stream_accepts_generator(detector):
    def gen():
        for n in (1, 2):
            yield make_images(n)

    results = list(detector.predict_stream(gen(), confidence=0.5, prefetch=1))
    assert [len(r) for r in results] == [1, 2]


def test_predict_stream_payload_passthrough(detector):
    def batches():
        yield "meta-a", make_images(2)
        yield make_images(1)  # bare batch mixes fine
        yield {"any": "payload"}, make_images(3)

    results = list(detector.predict_stream(batches(), confidence=0.5))

    assert results[0][0] == "meta-a" and len(results[0][1]) == 2
    assert len(results[1]) == 1  # bare batch yields bare predictions
    assert results[2][0] == {"any": "payload"} and len(results[2][1]) == 3


def test_predict_stream_propagates_source_errors(detector):
    def batches():
        yield make_images(1)
        raise OSError("source broke")

    stream = detector.predict_stream(batches(), confidence=0.5)
    with pytest.raises(OSError, match="source broke"):
        list(stream)


def test_predict_stream_empty_batch_yields_empty(detector):
    results = list(
        detector.predict_stream([("payload", []), make_images(1)], confidence=0.5)
    )
    assert results[0] == ("payload", [])
    assert len(results[1]) == 1


def test_image_loader_batches_and_pairs(detector):
    from orient_express.predictors import ImageLoader

    images = {f"item{i}": make_images(1)[0] for i in range(7)}
    loader = ImageLoader(list(images), load=lambda k: images[k], batch_size=3)

    results = list(detector.predict_stream(loader, confidence=0.5))

    items = [item for batch_items, _ in results for item in batch_items]
    assert items == [f"item{i}" for i in range(7)]
    assert [len(preds) for _, preds in results] == [3, 3, 1]
    for batch_items, preds in results:
        assert len(batch_items) == len(preds)


def test_image_loader_skips_failed_loads(detector):
    from orient_express.predictors import ImageLoader

    failures = []

    def load(item):
        if item == "bad":
            raise OSError("download failed")
        return make_images(1)[0]

    loader = ImageLoader(
        ["a", "bad", "b"],
        load=load,
        batch_size=2,
        on_error=lambda item, exc: failures.append(item),
    )
    results = list(detector.predict_stream(loader, confidence=0.5))

    items = [item for batch_items, _ in results for item in batch_items]
    assert items == ["a", "b"]
    assert failures == ["bad"]


def test_image_loader_pulls_lazily():
    from orient_express.predictors import ImageLoader

    pulled = []

    def items():
        for i in range(100):
            pulled.append(i)
            yield i

    loader = ImageLoader(items(), load=lambda i: make_images(1)[0], batch_size=2)
    next(iter(loader))
    # bounded by batch_size * (prefetch + 1) and workers, far below 100
    assert len(pulled) < 40


def test_infer_only_feeds_model_inputs(detector):
    images = make_images(1)
    feed = detector.preprocess(images)
    feed["extra_context"] = np.zeros(1)
    detector.infer(feed)
    sent = detector.session.run_inputs[-1]
    assert set(sent) == {"images", "target_sizes"}


def test_predict_stream_early_exit_unblocks_producer(detector):
    import threading
    import time

    baseline = threading.active_count()
    for _ in range(3):
        stream = detector.predict_stream(
            [make_images(1) for _ in range(20)], confidence=0.5
        )
        next(stream)
        stream.close()  # abandon the rest of the stream

    # the producer threads must unblock and exit, not park in queue.put()
    deadline = time.time() + 5.0
    while threading.active_count() > baseline and time.time() < deadline:
        time.sleep(0.01)
    assert threading.active_count() <= baseline


def test_predict_stream_prefetch_overrides_loader(detector):
    from orient_express.predictors import ImageLoader

    pulled = []

    def items():
        for i in range(100):
            pulled.append(i)
            yield i

    loader = ImageLoader(
        items(),
        load=lambda i: make_images(1)[0],
        batch_size=2,
        workers=1,
        prefetch=30,
    )
    stream = detector.predict_stream(loader, confidence=0.5, prefetch=1)
    next(stream)
    # the loader's own prefetch (30) would pull 60+ items ahead; the explicit
    # override caps the window at max(batch_size * (1 + 1), workers) = 4
    assert len(pulled) < 10
    stream.close()


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


def test_trt_implicit_profile_locks_first_shapes(detector):
    detector._trt_enforce_profile = True  # as if device="tensorrt", no profiles
    detector.predict(make_images(2), confidence=0.5)
    detector.predict(make_images(2), confidence=0.5)  # same shape: fine
    with pytest.raises(ValueError, match="engine rebuild"):
        detector.predict(make_images(3), confidence=0.5)


@pytest.mark.parametrize("bad_prefetch", [0, -1, 2.5, "2"])
def test_predict_stream_rejects_invalid_prefetch(detector, bad_prefetch):
    with pytest.raises(ValueError, match="prefetch"):
        next(
            detector.predict_stream(
                [make_images(1)], confidence=0.5, prefetch=bad_prefetch
            )
        )


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
            provider_options={"trt_profile_max_shapes": "images:8x64x64x3"},
            trt_enforce_profile=False,
        )
    from_dir.assert_called_once_with(
        str(tmp_path),
        metadata,
        "cpu",
        provider_options={"trt_profile_max_shapes": "images:8x64x64x3"},
        trt_enforce_profile=False,
    )


def test_get_predictor_joblib_rejects_kwargs(tmp_path):
    import yaml

    from orient_express.predictors import get_predictor
    from orient_express.utils.paths import get_metadata_path

    with open(get_metadata_path(str(tmp_path)), "w") as f:
        yaml.dump({"model_type": "joblib", "model_file": "m.joblib"}, f)
    with pytest.raises(TypeError, match="joblib"):
        get_predictor(str(tmp_path), provider_options={})
