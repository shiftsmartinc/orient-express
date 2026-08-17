"""Fault injection for Google Cloud Storage -- 12 artificial failures.

These are the calls where the retry policy is executed by GOOGLE'S client, not
by us: we hand `retry=` to `blob.exists()` and the library re-issues the HTTP
request underneath. A fake blob would therefore prove only that a policy was
passed, never that it fires. So the fault is injected at the transport: an
AuthorizedSession that returns synthetic 503s, 404s and dropped connections.

No network, no GCP project, no credentials -- `AnonymousCredentials` plus a
patched session is enough to drive the real retry machinery end to end.

Run standalone:  uv run python tests/test_retry_faults_gcs.py
Or with the suite: uv run pytest tests/test_retry_faults_gcs.py -v
"""

import base64
import io
import json
import logging
import sys
import types
from importlib.util import find_spec

import google_crc32c
import pytest
import requests
import urllib3
from google.api_core import exceptions as api_exceptions
from google.api_core.exceptions import RetryError
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage

from orient_express.utils import gs
from orient_express.utils.retry import get_gcs_retry_policy

# `orient_express.predictors` imports the ONNX runtime at module scope, and a
# dev install without an inference extra (`orient_express[cpu]`) does not have
# it. The one test below that reaches into the TRT cache syncer needs the class,
# not a model, so a placeholder is enough -- and these retry tests stay runnable
# in a bare environment.
if "onnxruntime" not in sys.modules and find_spec("onnxruntime") is None:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

PAYLOAD = b"payload"
# The client verifies the checksum it is told to expect, so the metadata has to
# describe the bytes the fake transport actually serves.
METADATA = {
    "name": "o",
    "bucket": "b",
    "size": str(len(PAYLOAD)),
    "contentType": "text/plain",
    "crc32c": base64.b64encode(google_crc32c.Checksum(PAYLOAD).digest()).decode(),
}


class FlakyTransport:
    """Stands in for AuthorizedSession.request.

    Fails the first `fail_times` requests -- with an HTTP status, or by raising
    `error` for the transport-level faults that have no status at all -- then
    serves a well-formed response.
    """

    def __init__(self, fail_times: int, status: int = 503, error: Exception = None):
        self.fail_times = fail_times
        self.status = status
        self.error = error
        self.attempts = 0

    def __call__(self, method, url, data=None, headers=None, **kwargs):
        if "projection=noAcl" in url:
            # The client looks the bucket up on its own before touching the
            # object. Serving it unconditionally keeps `attempts` a count of
            # the call under test rather than of the library's housekeeping.
            return self._response(200, json.dumps({"name": "b"}).encode())
        self.attempts += 1
        if self.attempts <= self.fail_times:
            if self.error is not None:
                raise self.error
            return self._response(self.status, b'{"error": {"message": "injected"}}')
        if method == "POST" and "uploadType=resumable" in url:
            # Resumable uploads open a session first; the client follows the
            # Location header for the chunks that carry the bytes.
            response = self._response(200, b"{}")
            response.headers["location"] = "https://upload.example/session/1"
            return response
        if "alt=media" in url:
            return self._media_response()
        return self._response(200, json.dumps(METADATA).encode())

    def _response(self, status, body, content_type="application/json"):
        response = requests.Response()
        response.status_code = status
        response._content = body
        response.headers["Content-Type"] = content_type
        response.request = requests.Request(method="GET", url="https://x/").prepare()
        return response

    def _media_response(self):
        # A byte download reads `response.raw`, which must be a real urllib3
        # response -- the checksum decoder asks it for its headers.
        response = self._response(200, PAYLOAD, content_type="text/plain")
        response.raw = urllib3.response.HTTPResponse(
            body=io.BytesIO(PAYLOAD),
            headers={"content-length": str(len(PAYLOAD))},
            status=200,
            preload_content=False,
        )
        response.headers["Content-Length"] = str(len(PAYLOAD))
        return response


@pytest.fixture
def inject(monkeypatch):
    """Point every storage.Client() in the package at a flaky transport."""

    def _inject(fail_times=0, status=503, error=None):
        transport = FlakyTransport(fail_times, status, error)
        # Built before the patch: constructing it afterwards would recurse.
        client = storage.Client(project="p", credentials=AnonymousCredentials())
        client._http.request = transport
        monkeypatch.setattr(storage, "Client", lambda *a, **k: client)
        return transport

    return _inject


@pytest.fixture
def short_budget(monkeypatch):
    """Shrink the 120s budget so the outage test finishes in about a second.

    Not smaller: the first backoff is 1s, so a budget under that would give up
    after a single attempt and prove nothing about retrying.
    """
    monkeypatch.setattr(gs, "get_gcs_retry_policy", lambda: get_gcs_retry_policy(2.5))


# --------------------------------------------------------------- transient


def test_exists_retries_a_transient_503_then_succeeds(inject):
    transport = inject(fail_times=2)
    assert gs.exists("gs://b/o") is True
    assert transport.attempts > 1, "the 503 must have been re-attempted"


def test_read_file_bytes_retries_then_returns_the_content(inject):
    transport = inject(fail_times=2)
    assert gs.read_file_bytes("gs://b/o") == PAYLOAD
    assert transport.attempts > 1


def test_download_file_retries_then_writes_the_file(inject, tmp_path):
    transport = inject(fail_times=2)
    target = tmp_path / "artifact.bin"
    gs.download_file("gs://b/o", str(target))
    assert target.read_bytes() == PAYLOAD
    assert transport.attempts > 1


def test_upload_file_retries_then_completes(inject, tmp_path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(PAYLOAD)
    transport = inject(fail_times=2)
    gs.upload_file(str(source), "gs://b/o")
    assert transport.attempts > 1


def test_rate_limit_429_is_retried(inject):
    transport = inject(fail_times=2, status=429)
    assert gs.exists("gs://b/o") is True
    assert transport.attempts > 1


# ------------------------------------------- the errors the old policy missed


def test_a_dropped_connection_is_retried(inject):
    """Regression guard.

    utils.gs used to hand-roll its own policy, which retried only on api_core
    status exceptions. A dropped connection carries no status, so it was NOT
    retried -- and it is one of the two commonest ways a GCS transfer dies.
    """
    transport = inject(fail_times=2, error=ConnectionError("connection reset"))
    assert gs.exists("gs://b/o") is True
    assert transport.attempts > 1


def test_a_read_timeout_is_retried(inject):
    """The other one the hand-rolled policy missed."""
    transport = inject(
        fail_times=2, error=requests.exceptions.Timeout("read timed out")
    )
    assert gs.exists("gs://b/o") is True
    assert transport.attempts > 1


# --------------------------------------------------------------- permanent


def test_a_404_is_not_retried(inject):
    """An absent object is an answer, not a failure -- one attempt, then False."""
    transport = inject(fail_times=99, status=404)
    assert gs.exists("gs://b/o") is False
    assert transport.attempts == 1


def test_a_403_is_not_retried(inject):
    transport = inject(fail_times=99, status=403)
    with pytest.raises(api_exceptions.Forbidden):
        gs.exists("gs://b/o")
    assert transport.attempts == 1


# --------------------------------------------------------------- budget


def test_a_sustained_outage_gives_up_rather_than_hanging(inject, short_budget):
    transport = inject(fail_times=999)
    with pytest.raises((RetryError, api_exceptions.ServiceUnavailable)):
        gs.exists("gs://b/o")
    assert transport.attempts > 1, "it should have tried more than once first"


def test_a_bounded_policy_stops_inside_the_callers_deadline():
    """What the TRT cache syncer relies on: its own timeout caps the retries.

    Without it a cold start would stall for the full 120s default on every
    worker, regardless of ORIENT_EXPRESS_TRT_CACHE_TIMEOUT.
    """
    from orient_express.predictors.runtime import _TrtCacheGcsSync

    syncer = _TrtCacheGcsSync.__new__(_TrtCacheGcsSync)
    syncer._timeout = 0.2
    policy = syncer._bounded_retry_policy()
    assert policy._timeout == 0.2

    attempts = []

    def always_unavailable():
        attempts.append(1)
        raise api_exceptions.ServiceUnavailable("503")

    with pytest.raises(RetryError):
        policy(always_unavailable)()
    assert attempts, "it must attempt at least once before giving up"


# --------------------------------------------------------------- logging


def test_retried_transfers_are_logged_as_gcs(inject, caplog):
    inject(fail_times=2)
    with caplog.at_level(logging.WARNING):
        assert gs.exists("gs://b/o") is True
    retried = [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()]
    assert retried, "a retried transfer must leave a trace"
    assert all("gcs" in message for message in retried)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
