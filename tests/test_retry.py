"""Retry behaviour, per transport.

Two policies, two contracts:

* `vertex.VERTEX_RETRY` (google.api_core) — retries on a predicate, so transient
  gRPC statuses are re-attempted and anything else fails on the first try with
  its original exception type.
* `utils.retry.retry` (house decorator) — best-effort, retries on any exception,
  but must still re-raise the original so callers see the real failure.

The predicate half is the point of the whole policy: a genuinely missing model
has to fail fast rather than spend the backoff budget, and the caller has to be
able to tell a 503 from a 404.
"""

import logging
import time

import pytest
from google.api_core import exceptions as api_exceptions

from orient_express.utils.retry import retry
from orient_express.vertex import GCS_RETRY, VERTEX_RETRY, _download_blob


class TestVertexRetry:
    """google.api_core policy applied to the Vertex control-plane calls."""

    def test_transient_error_is_retried_then_succeeds(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise api_exceptions.ServiceUnavailable("503 unavailable")
            return "ok"

        assert VERTEX_RETRY(flaky)() == "ok"
        assert len(attempts) == 3

    @pytest.mark.parametrize(
        "error",
        [
            api_exceptions.ServiceUnavailable("503"),
            api_exceptions.DeadlineExceeded("504"),
            api_exceptions.InternalServerError("500"),
            api_exceptions.Aborted("409"),
            api_exceptions.TooManyRequests("429"),
        ],
    )
    def test_every_transient_status_in_the_predicate_is_retried(self, error):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise error
            return "ok"

        assert VERTEX_RETRY(flaky)() == "ok"
        assert len(attempts) == 2

    @pytest.mark.parametrize(
        "error",
        [
            api_exceptions.NotFound("no such model"),
            api_exceptions.PermissionDenied("403"),
            api_exceptions.InvalidArgument("400"),
            ValueError("not a google error at all"),
        ],
    )
    def test_permanent_error_fails_immediately_with_its_own_type(self, error):
        """The criterion this policy exists for: no retry, no type laundering.

        A wrapped exception would leave get_vertex_model unable to distinguish a
        transient outage from a model that simply is not there.
        """
        attempts = []

        def boom():
            attempts.append(1)
            raise error

        with pytest.raises(type(error)):
            VERTEX_RETRY(boom)()
        assert len(attempts) == 1

    def test_budget_is_bounded_rather_than_infinite(self):
        assert VERTEX_RETRY._timeout == 60.0
        assert VERTEX_RETRY._initial == 1.0
        assert VERTEX_RETRY._maximum == 10.0


class TestHouseRetryDecorator:
    """utils.retry — best-effort, but must not swallow the failure."""

    def test_retries_then_succeeds(self):
        attempts = []

        @retry(retries=3, initial_timeout=0.01, max_timeout=0.01)
        def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise RuntimeError("transient")
            return "ok"

        assert flaky() == "ok"
        assert len(attempts) == 2

    def test_reraises_the_original_exception_not_a_wrapper(self):
        """Regression guard on the wrapper.

        It used to raise a bare Exception after exhausting retries, discarding
        both the original type and the traceback.
        """
        attempts = []

        @retry(retries=3, initial_timeout=0.01, max_timeout=0.01)
        def boom():
            attempts.append(1)
            raise FileNotFoundError("gone")

        with pytest.raises(FileNotFoundError, match="gone"):
            boom()
        assert len(attempts) == 3

    def test_does_not_sleep_after_the_final_attempt(self):
        """3 attempts means 2 gaps. A trailing sleep only delays the failure."""

        @retry(retries=3, initial_timeout=0.1, max_timeout=0.1)
        def always_fails():
            raise ZeroDivisionError

        started = time.perf_counter()
        with pytest.raises(ZeroDivisionError):
            always_fails()
        elapsed = time.perf_counter() - started
        assert 0.15 < elapsed < 0.35, f"expected ~0.2s for two gaps, got {elapsed:.2f}s"

    def test_backoff_is_capped(self):
        delays = []

        @retry(retries=4, initial_timeout=1.0, max_timeout=2.0)
        def always_fails():
            raise RuntimeError

        with pytest.raises(RuntimeError):
            with_patched_sleep(delays, always_fails)
        # 1.0 -> 2.0 -> capped at 2.0, and no sleep after the last attempt
        assert delays == [1.0, 2.0, 2.0]


def with_patched_sleep(recorded, func):
    """Run func with time.sleep recording instead of sleeping."""
    import orient_express.utils.retry as retry_module

    original = retry_module.time.sleep
    retry_module.time.sleep = recorded.append
    try:
        return func()
    finally:
        retry_module.time.sleep = original


class TestRetriesAreLogged:
    """A silent retry is nearly as bad as no retry.

    The incident that opened this issue was invisible until a pipeline failed. If
    retries succeed without a trace, the next wobble looks like the first one --
    so each transport has to leave a record, and each mechanism differs: Vertex
    gets `on_error`, GCS gets a logging predicate because api_core exposes no
    `with_on_error` for a copy of Google's own policy.
    """

    def test_vertex_retry_logs_each_retried_attempt(self, caplog):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise api_exceptions.ServiceUnavailable("503 unavailable")
            return "ok"

        with caplog.at_level(logging.WARNING):
            assert VERTEX_RETRY(flaky)() == "ok"

        retried = [r for r in caplog.records if "retrying" in r.getMessage()]
        assert len(retried) == 2, "one warning per retried attempt, not per call"
        assert "vertex" in retried[0].getMessage()
        assert "ServiceUnavailable" in retried[0].getMessage()

    def test_vertex_retry_does_not_log_a_clean_call(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert VERTEX_RETRY(lambda: "ok")() == "ok"
        assert not [r for r in caplog.records if "retrying" in r.getMessage()]

    def test_vertex_retry_does_not_log_a_permanent_failure_as_a_retry(self, caplog):
        """A 404 is the caller's to report; it was never retried."""

        def boom():
            raise api_exceptions.NotFound("no such model")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(api_exceptions.NotFound):
                VERTEX_RETRY(boom)()
        assert not [r for r in caplog.records if "retrying" in r.getMessage()]

    def test_gcs_retry_logs_and_still_answers_the_predicate(self, caplog):
        """The logging wrapper must not change WHICH errors are retriable."""
        transient = api_exceptions.ServiceUnavailable("503")
        permanent = api_exceptions.NotFound("gone")

        with caplog.at_level(logging.WARNING):
            assert GCS_RETRY._predicate(transient) is True
            assert GCS_RETRY._predicate(permanent) is False

        messages = [
            r.getMessage() for r in caplog.records if "retrying" in r.getMessage()
        ]
        assert len(messages) == 1, "only the retriable error is logged"
        assert "gcs" in messages[0]

    def test_gcs_retry_keeps_googles_backoff(self):
        """Only logging was added -- the timing and error set stay Google's."""
        from google.cloud.storage.retry import DEFAULT_RETRY

        assert GCS_RETRY._initial == DEFAULT_RETRY._initial
        assert GCS_RETRY._maximum == DEFAULT_RETRY._maximum
        assert GCS_RETRY._multiplier == DEFAULT_RETRY._multiplier
        assert GCS_RETRY._timeout == DEFAULT_RETRY._timeout


class TestDownloadBlobRetry:
    """The >8MB chunked branch was the one wholly unprotected path.

    transfer_manager.download_chunks_concurrently takes no `retry` argument, so the
    policy has to wrap _download_blob itself. These assert the wrapper is actually
    there and re-attempts the WHOLE transfer -- the thing the issue called out.
    """

    def test_chunked_download_is_retried_as_a_whole(self, monkeypatch, tmp_path):
        blob = type("Blob", (), {"size": 64 * 1024 * 1024, "name": "big.onnx"})()
        calls = []

        def flaky_chunked(*args, **kwargs):
            calls.append(1)
            if len(calls) < 2:
                raise api_exceptions.ServiceUnavailable("503 unavailable")

        monkeypatch.setattr(
            "orient_express.vertex.transfer_manager.download_chunks_concurrently",
            flaky_chunked,
        )
        _download_blob(blob, str(tmp_path / "big.onnx"))
        assert len(calls) == 2, "the chunked transfer must be re-attempted"

    def test_small_download_is_retried(self, monkeypatch, tmp_path):
        calls = []

        class Blob:
            size = 1024
            name = "small.json"

            def download_to_filename(self, path, **kwargs):
                calls.append(1)
                if len(calls) < 2:
                    raise api_exceptions.ServiceUnavailable("503 unavailable")

        _download_blob(Blob(), str(tmp_path / "small.json"))
        assert len(calls) == 2

    def test_permanent_download_error_is_not_retried(self, monkeypatch, tmp_path):
        calls = []

        class Blob:
            size = 1024
            name = "gone.json"

            def download_to_filename(self, path, **kwargs):
                calls.append(1)
                raise api_exceptions.NotFound("no such object")

        with pytest.raises(api_exceptions.NotFound):
            _download_blob(Blob(), str(tmp_path / "gone.json"))
        assert len(calls) == 1, "a missing object must fail fast"
