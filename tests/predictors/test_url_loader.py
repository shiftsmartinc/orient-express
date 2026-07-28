"""Tests for UrlImageLoader against a local HTTP server (no network)."""

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from orient_express.predictors import UrlImageLoader
from orient_express.predictors.object_detection import BoundingBoxPredictor

RESOLUTION = 64


def jpeg_bytes(seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture(scope="module")
def image_server():
    images = {f"/img/{i}.jpg": jpeg_bytes(i) for i in range(24)}
    truncated = jpeg_bytes(99)
    flaky_body = jpeg_bytes(50)
    flaky_hits: dict[str, int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/private/"):
                if self.headers.get("Authorization") != "Bearer tok":
                    self.send_error(401)
                    return
                body = flaky_body
            elif self.path.startswith("/flaky/"):
                # 503 on the first hit of each path, succeed after
                flaky_hits[self.path] = flaky_hits.get(self.path, 0) + 1
                if flaky_hits[self.path] == 1:
                    self.send_error(503)
                    return
                body = flaky_body
            elif self.path == "/missing.jpg":
                self.send_error(404)
                return
            elif self.path == "/empty.jpg":
                body = b""
            elif self.path == "/truncated.jpg":
                body = truncated[: len(truncated) // 2]
            else:
                body = images.get(self.path)
                if body is None:
                    self.send_error(404)
                    return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_batches_ordered_and_complete(image_server):
    refs = [f"{image_server}/img/{i}.jpg" for i in range(24)]
    loader = UrlImageLoader(refs, batch_size=5, concurrency=8)

    seen = []
    sizes = []
    for payload, images in loader:
        assert len(payload) == len(images)
        assert all(img.size == (160, 120) for img in images)
        seen.extend(payload)
        sizes.append(len(images))
    assert seen == refs  # input order, nothing lost
    assert sizes == [5, 5, 5, 5, 4]


def test_bad_items_skipped_and_reported(image_server, caplog):
    import logging

    refs = [f"{image_server}/img/{i}.jpg" for i in range(6)]
    refs.insert(2, f"{image_server}/missing.jpg")  # 404: skipped
    refs.insert(5, f"{image_server}/truncated.jpg")  # processed, with warning
    refs.insert(7, f"{image_server}/empty.jpg")  # zero bytes: skipped
    failures = []
    loader = UrlImageLoader(
        refs,
        batch_size=4,
        concurrency=4,
        on_error=lambda item, exc: failures.append(item.rsplit("/", 1)[1]),
    )

    with caplog.at_level(logging.WARNING):
        seen = [item for payload, _ in loader for item in payload]
    # the truncated image is decoded (readable rows + gray fill) and kept;
    # only the items that yield no pixels at all are skipped
    assert len(seen) == 7
    assert f"{image_server}/truncated.jpg" in seen
    assert sorted(failures) == ["empty.jpg", "missing.jpg"]
    assert any("truncated" in r.message for r in caplog.records)


def test_url_callable_and_payload(image_server):
    rows = [{"id": i, "url": f"{image_server}/img/{i}.jpg"} for i in range(7)]
    loader = UrlImageLoader(rows, url=lambda r: r["url"], batch_size=3)
    seen = [row["id"] for payload, _ in loader for row in payload]
    assert seen == list(range(7))


def test_non_url_items_error_per_item(image_server):
    # a gs:// URI is not a URL; it fails naturally in the fetch and is
    # reported like any other bad item, without special handling
    refs = [f"{image_server}/img/0.jpg", "gs://bucket/img.jpg"]
    failures = []
    loader = UrlImageLoader(refs, on_error=lambda item, exc: failures.append(item))
    seen = [item for payload, _ in loader for item in payload]
    assert seen == [f"{image_server}/img/0.jpg"]
    assert failures == ["gs://bucket/img.jpg"]


def test_custom_headers_sent(image_server):
    # `headers` is the auth mechanism: the /private/ path 401s without the
    # right Authorization header
    refs = [f"{image_server}/private/a.jpg"]
    failures = []
    loader = UrlImageLoader(refs, on_error=lambda item, exc: failures.append(item))
    assert [item for payload, _ in loader for item in payload] == []
    assert failures == refs  # unauthenticated: rejected

    loader = UrlImageLoader(refs, headers={"Authorization": "Bearer tok"})
    assert [item for payload, _ in loader for item in payload] == refs


def test_fast_decode_requires_target_outside_predict_stream(image_server):
    loader = UrlImageLoader([f"{image_server}/img/0.jpg"], batch_size=1, decode="fast")
    with pytest.raises(ValueError, match="fast_size"):
        next(iter(loader))


@pytest.fixture
def detector(mock_onnx_session, class_mapping):
    session = mock_onnx_session(
        resolution=RESOLUTION,
        input_names=["images", "target_sizes"],
        output_names=["boxes", "scores", "labels"],
    )

    def run(output_names, input_dict):
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
        yield BoundingBoxPredictor("fake.onnx", {1: "thing"})


def test_predict_stream_fuses_with_url_loader(image_server, detector):
    refs = [f"{image_server}/img/{i}.jpg" for i in range(10)]
    loader = UrlImageLoader(refs, batch_size=4, concurrency=4)

    results = list(detector.predict_stream(loader, confidence=0.5))

    seen = [item for payload, _ in results for item in payload]
    assert seen == refs
    assert [len(preds) for _, preds in results] == [4, 4, 2]


def test_early_exit_cleans_up_threads(image_server, detector):
    import time

    baseline = threading.active_count()
    for _ in range(3):
        loader = UrlImageLoader(
            [f"{image_server}/img/{i % 24}.jpg" for i in range(100)],
            batch_size=4,
            concurrency=8,
        )
        stream = detector.predict_stream(loader, confidence=0.5)
        next(stream)
        stream.close()

    deadline = time.time() + 5
    while threading.active_count() > baseline and time.time() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= baseline


def test_transient_errors_retried(image_server):
    # object stores throw occasional 429/503s at high concurrency; a
    # transient failure must cost a retry, not the item
    refs = [f"{image_server}/img/{i}.jpg" for i in range(3)]
    refs.insert(1, f"{image_server}/flaky/retried.jpg")  # 503s once
    loader = UrlImageLoader(refs, batch_size=2, retry_backoff=0.01)
    seen = [item for payload, _ in loader for item in payload]
    assert seen == refs  # nothing dropped, order kept


def test_retries_zero_reports_transient_failures(image_server):
    refs = [f"{image_server}/img/{i}.jpg" for i in range(3)]
    refs.insert(1, f"{image_server}/flaky/dropped.jpg")  # 503s once
    failures = []
    loader = UrlImageLoader(
        refs,
        batch_size=2,
        retries=0,
        on_error=lambda item, exc: failures.append(item),
    )
    seen = [item for payload, _ in loader for item in payload]
    assert len(seen) == 3
    assert failures == [f"{image_server}/flaky/dropped.jpg"]
