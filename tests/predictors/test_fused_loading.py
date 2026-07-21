"""Tests for fused ImageLoader loading and the stream utilities."""

import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors import (
    BoundingBoxPredictor,
    ClassificationPredictor,
    ImageLoader,
    SemanticSegmentationPredictor,
    flat_map_stream,
    map_stream,
)

RESOLUTION = 64


def make_images(n, sizes=((80, 60), (120, 90), (64, 64))):
    rng = np.random.default_rng(0)
    return [
        Image.fromarray(
            rng.integers(
                0,
                255,
                (sizes[i % len(sizes)][1], sizes[i % len(sizes)][0], 3),
                dtype=np.uint8,
            )
        )
        for i in range(n)
    ]


@pytest.mark.parametrize(
    ("cls", "input_names", "output_names"),
    [
        (
            BoundingBoxPredictor,
            ["images", "target_sizes"],
            ["boxes", "scores", "labels"],
        ),
        (ClassificationPredictor, ["images"], ["scores"]),
        (SemanticSegmentationPredictor, ["images"], ["masks"]),
    ],
)
def test_assemble_feed_matches_preprocess(
    mock_onnx_session, class_mapping, cls, input_names, output_names
):
    session = mock_onnx_session(
        resolution=RESOLUTION, input_names=input_names, output_names=output_names
    )
    with patch(
        "orient_express.predictors.runtime.ort.InferenceSession",
        return_value=session,
    ):
        predictor = cls("fake.onnx", class_mapping)

    images = make_images(3)
    via_preprocess = predictor.preprocess(images)
    arrays, sizes = zip(
        *(predictor.preprocess_item(img) for img in images), strict=True
    )
    via_assemble = predictor.assemble_feed(list(arrays), list(sizes))

    assert set(via_preprocess) == set(via_assemble)
    for key in via_preprocess:
        np.testing.assert_array_equal(via_preprocess[key], via_assemble[key])


@pytest.fixture
def detector(mock_onnx_session, class_mapping):
    session = mock_onnx_session(
        resolution=RESOLUTION,
        input_names=["images", "target_sizes"],
        output_names=["boxes", "scores", "labels"],
    )

    def run(output_names, input_dict):
        session.run_inputs.append(input_dict)
        n = len(input_dict["images"])
        return [
            np.tile([10.0, 10.0, 50.0, 50.0], (n, 1, 1)),
            np.full((n, 1), 0.9),
            np.ones((n, 1), dtype=np.int64),
        ]

    session.run.side_effect = run
    with patch(
        "orient_express.predictors.runtime.ort.InferenceSession",
        return_value=session,
    ):
        yield BoundingBoxPredictor("fake.onnx", class_mapping)


def test_fused_stream_matches_predict(detector):
    images = {f"item{i}": img for i, img in enumerate(make_images(7))}
    loader = ImageLoader(list(images), load=lambda k: images[k], batch_size=3)

    results = list(detector.predict_stream(loader, confidence=0.5))

    direct = detector.predict(list(images.values()), confidence=0.5)
    streamed_flat = [
        p for _, preds in results for batch_preds in [preds] for p in batch_preds
    ]
    assert [item for items, _ in results for item in items] == list(images)
    for s, d in zip(streamed_flat, direct, strict=True):
        assert [x.to_dict() for x in s] == [x.to_dict() for x in d]


def test_fused_stream_keep_original(detector):
    images = make_images(4)
    loader = ImageLoader(
        list(range(4)), load=lambda i: images[i], batch_size=2, keep_original=True
    )
    results = list(detector.predict_stream(loader, confidence=0.5))

    for payload, preds in results:
        assert len(payload) == len(preds)
        for (item, original), _ in zip(payload, preds, strict=True):
            assert original is images[item]


def test_fused_stream_skips_failed_loads(detector):
    failures = []

    def load(item):
        if item == "bad":
            raise OSError("boom")
        return make_images(1)[0]

    loader = ImageLoader(
        ["a", "bad", "b"],
        load=load,
        batch_size=2,
        on_error=lambda item, exc: failures.append(item),
    )
    results = list(detector.predict_stream(loader, confidence=0.5))
    assert [i for items, _ in results for i in items] == ["a", "b"]
    assert failures == ["bad"]


def test_map_stream_ordered_and_threaded():
    out = list(map_stream(lambda x: x * 2, range(20), workers=4))
    assert out == [x * 2 for x in range(20)]


def test_map_stream_propagates_errors():
    def fn(x):
        if x == 3:
            raise ValueError("bad item")
        return x

    with pytest.raises(ValueError, match="bad item"):
        list(map_stream(fn, range(10), workers=4))


def test_flat_map_stream_expands():
    out = list(flat_map_stream(lambda x: [x] * x, [1, 2, 3], workers=2))
    assert out == [1, 2, 2, 3, 3, 3]


def test_map_stream_is_lazy():
    pulled = []

    def items():
        for i in range(100):
            pulled.append(i)
            yield i

    stream = map_stream(lambda x: x, items(), workers=2, prefetch=4)
    next(stream)
    assert len(pulled) < 20


def make_truncated_image():
    """A PIL image whose lazy decode raises, like a truncated download."""
    buf = io.BytesIO()
    make_images(1, sizes=((200, 200),))[0].save(buf, "JPEG")
    data = buf.getvalue()
    return Image.open(io.BytesIO(data[: len(data) // 2]))


def test_fused_stream_skips_corrupt_decodes(detector):
    failures = []
    good = make_images(2)
    images = {"a": good[0], "bad": make_truncated_image(), "b": good[1]}
    loader = ImageLoader(
        list(images),
        load=lambda k: images[k],
        batch_size=2,
        on_error=lambda item, exc: failures.append(item),
    )
    results = list(detector.predict_stream(loader, confidence=0.5))
    assert [i for items, _ in results for i in items] == ["a", "b"]
    assert failures == ["bad"]


def test_loader_iteration_skips_corrupt_decodes(detector):
    failures = []
    images = {"a": make_images(1)[0], "bad": make_truncated_image()}
    loader = ImageLoader(
        list(images),
        load=lambda k: images[k],
        batch_size=2,
        on_error=lambda item, exc: failures.append(item),
    )
    # __iter__ (the generic, non-fused source) must also force the decode
    batches = list(loader)
    assert [i for items, _ in batches for i in items] == ["a"]
    assert failures == ["bad"]
