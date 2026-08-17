"""Fault injection for the two HTTP paths -- 12 artificial failures.

Neither of these uses a google.api_core policy, because neither talks to a
Google API:

* `utils.retry.retry`, the house decorator, guards `serving.download_image`.
  Best-effort by design -- it retries ANY exception, because the caller only
  wants the image and has no use for the distinction.
* `UrlImageLoader` runs its own async loop, because api_core is synchronous and
  knows nothing about aiohttp's exception types. It still honours the same
  contract: a predicate, exponential backoff, a logged attempt.

Both run their retry loop in OUR code, so the fault goes straight into the
callable -- no transport patching needed, unlike the GCS script.

Run standalone:  uv run python tests/test_retry_faults_http.py
Or with the suite: uv run pytest tests/test_retry_faults_http.py -v
"""

import logging
import sys
import types
from importlib.util import find_spec

import aiohttp
import pytest

from orient_express import serving
from orient_express.utils.retry import retry

# `orient_express.predictors` imports the ONNX runtime at module scope, and a
# dev install without an inference extra (`orient_express[cpu]`) does not have
# it. Nothing here runs a model -- these tests only need the loader's fetch
# stage -- so a placeholder is enough to get the import through, and the retry
# tests stay runnable in a bare environment.
if "onnxruntime" not in sys.modules and find_spec("onnxruntime") is None:
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

PAYLOAD = b"\x89PNG fake image bytes"


def record_sleeps(monkeypatch):
    """Capture the decorator's backoff instead of waiting it out."""
    import orient_express.utils.retry as retry_module

    delays = []
    monkeypatch.setattr(retry_module.time, "sleep", delays.append)
    return delays


# =========================================================== house decorator


def test_decorator_retries_a_transient_failure_then_succeeds(monkeypatch):
    record_sleeps(monkeypatch)
    attempts = []

    @retry(retries=3)
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("connection reset by peer")
        return "image"

    assert flaky() == "image"
    assert len(attempts) == 3


def test_decorator_exhausts_its_attempts_and_reraises_the_original_type(monkeypatch):
    """Best-effort does not mean silent: the real failure must reach the caller."""
    record_sleeps(monkeypatch)
    attempts = []

    @retry(retries=3)
    def always_fails():
        attempts.append(1)
        raise FileNotFoundError("no such image")

    with pytest.raises(FileNotFoundError, match="no such image"):
        always_fails()
    assert len(attempts) == 3


def test_decorator_retries_any_exception_type(monkeypatch):
    """The deliberate difference from the predicate policies.

    Fetching an image has no permanent-vs-transient distinction worth making --
    the caller just wants the bytes -- so everything is worth one more try.
    """
    record_sleeps(monkeypatch)

    def failing_once_with(error):
        """A factory, so each closure binds its own error rather than the last."""
        attempts = []

        @retry(retries=2)
        def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise error
            return "image"

        return flaky, attempts

    for error in (ValueError("bad"), OSError("disk"), RuntimeError("boom")):
        flaky, attempts = failing_once_with(error)
        assert flaky() == "image"
        assert len(attempts) == 2, f"{type(error).__name__} was not retried"


def test_decorator_backoff_doubles_and_is_capped(monkeypatch):
    delays = record_sleeps(monkeypatch)

    @retry(retries=5, initial_timeout=1.0, max_timeout=2.0)
    def always_fails():
        raise RuntimeError

    with pytest.raises(RuntimeError):
        always_fails()
    assert delays == [1.0, 2.0, 2.0, 2.0], "doubling, then held at the cap"


def test_decorator_does_not_sleep_after_the_final_attempt(monkeypatch):
    """4 attempts means 3 gaps. A trailing sleep only delays the failure."""
    delays = record_sleeps(monkeypatch)

    @retry(retries=4, initial_timeout=0.5)
    def always_fails():
        raise RuntimeError

    with pytest.raises(RuntimeError):
        always_fails()
    assert len(delays) == 3


def test_download_image_call_site_is_covered_and_logged(monkeypatch, caplog):
    """The real entry point, not just the bare decorator."""
    record_sleeps(monkeypatch)
    attempts = []

    def flaky_read(address, session=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("connection reset by peer")
        return PAYLOAD

    monkeypatch.setattr(serving, "read_image_from_url", flaky_read)
    with caplog.at_level(logging.WARNING):
        assert serving.download_image("https://example/img.png") == PAYLOAD

    assert len(attempts) == 3
    retried = [r for r in caplog.records if "retrying" in r.getMessage()]
    assert len(retried) == 2, "one warning per retried attempt"


# ================================================================ url loader


class FakeResponse:
    """One planned outcome: raise on entry, or serve a status and a body."""

    def __init__(self, status=200, body=PAYLOAD, error=None):
        self.status = status
        self.body = body
        self.error = error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=None, history=(), status=self.status
            )

    async def read(self):
        return self.body


class FakeSession:
    """Stands in for aiohttp.ClientSession; `plan(url, attempt)` decides each try."""

    def __init__(self, plan):
        self.plan = plan
        self.attempts = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def get(self, url, headers=None):
        self.attempts[url] = self.attempts.get(url, 0) + 1
        return self.plan(url, self.attempts[url])


@pytest.fixture
def fetch(monkeypatch):
    """Drive UrlImageLoader's fetch stage against a planned fake session."""

    def _fetch(plan, items=("https://example/a.png",), **kwargs):
        # Imported here so the stub above is in place first.
        from orient_express.predictors.loader import UrlImageLoader

        session = FakeSession(plan)
        monkeypatch.setattr(aiohttp, "TCPConnector", lambda **kw: None)
        monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: session)
        errors = []
        loader = UrlImageLoader(
            list(items),
            retries=kwargs.pop("retries", 2),
            retry_backoff=kwargs.pop("retry_backoff", 0.001),
            concurrency=2,
            on_error=lambda item, exc: errors.append((item, exc)),
            **kwargs,
        )
        return list(loader._byte_stream(4)), errors, session

    return _fetch


def test_loader_retries_a_transient_status_then_returns_the_bytes(fetch):
    def plan(url, attempt):
        return FakeResponse(status=503) if attempt < 3 else FakeResponse()

    results, errors, session = fetch(plan)
    assert results == [("https://example/a.png", PAYLOAD)]
    assert session.attempts["https://example/a.png"] == 3
    assert not errors


def test_loader_does_not_retry_a_404(fetch):
    """A missing image is permanent -- one attempt, then skip the item."""

    def plan(url, attempt):
        return FakeResponse(status=404)

    results, errors, session = fetch(plan)
    assert session.attempts["https://example/a.png"] == 1
    assert results == [("https://example/a.png", None)]
    assert len(errors) == 1


def test_loader_retries_a_dropped_connection(fetch):
    def plan(url, attempt):
        if attempt < 2:
            return FakeResponse(error=aiohttp.ClientConnectionError("reset"))
        return FakeResponse()

    results, errors, session = fetch(plan)
    assert results == [("https://example/a.png", PAYLOAD)]
    assert session.attempts["https://example/a.png"] == 2


def test_loader_retries_a_timeout(fetch):
    """Object stores throw these at a few hundred concurrent requests."""

    def plan(url, attempt):
        return FakeResponse(error=TimeoutError()) if attempt < 2 else FakeResponse()

    results, errors, session = fetch(plan)
    assert results == [("https://example/a.png", PAYLOAD)]
    assert session.attempts["https://example/a.png"] == 2


def test_loader_reports_an_exhausted_item_and_keeps_the_others(fetch):
    """One dead URL must not take the batch down with it."""
    dead, alive = "https://example/dead.png", "https://example/alive.png"

    def plan(url, attempt):
        return FakeResponse(status=503) if url == dead else FakeResponse()

    results, errors, session = fetch(plan, items=(dead, alive), retries=2)
    assert dict(results) == {dead: None, alive: PAYLOAD}
    assert session.attempts[dead] == 3, "the initial attempt plus two retries"
    assert [item for item, _ in errors] == [dead]


def test_loader_logs_each_retried_fetch(fetch, caplog):
    def plan(url, attempt):
        return FakeResponse(status=503) if attempt < 3 else FakeResponse()

    with caplog.at_level(logging.WARNING):
        results, _, _ = fetch(plan)
    assert results == [("https://example/a.png", PAYLOAD)]
    retried = [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()]
    assert len(retried) == 2
    assert all("http" in message for message in retried)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
