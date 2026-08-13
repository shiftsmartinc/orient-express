"""Retry behaviour, per transport.

Three mechanisms, all built or shaped by `utils.retry`:

* `VERTEX_RETRY` / `GCS_RETRY` (google.api_core) — retry on a predicate, so
  transient statuses are re-attempted and anything else fails on the first try
  with its original exception type.
* `utils.retry.retry` (house decorator) — best-effort, retries on any exception,
  but must still re-raise the original so callers see the real failure.
* `UrlImageLoader`'s own async loop — hand-rolled for aiohttp, but honouring the
  same contract: a predicate, exponential backoff, a logged attempt.

The predicate half is the point of the whole policy: a genuinely missing model
has to fail fast rather than spend the backoff budget, and the caller has to be
able to tell a 503 from a 404. `TestPoliciesAreBuiltInOnePlace` guards the other
half — that no module quietly grows a fourth policy of its own.
"""

import logging
import time

import pytest
from google.api_core import exceptions as api_exceptions

from orient_express.utils.retry import (
    get_gcs_retry_policy,
    get_vertex_retry_policy,
    retry,
)
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


class TestGsHelpersAreRetried:
    """utils/gs.py had two calls with no policy while its neighbours had one.

    `download_file` retried and `read_file_bytes` did not, in the same module --
    the same coverage-by-accident the Vertex side was filed for.
    """

    def test_exists_passes_a_retry(self, monkeypatch):
        """A 503 must not be reported to the caller as "the object is absent"."""
        from orient_express.utils import gs

        seen = {}

        class Blob:
            def exists(self, retry=None):
                seen["retry"] = retry
                return True

        class Bucket:
            def blob(self, path):
                return Blob()

        monkeypatch.setattr(
            gs.storage,
            "Client",
            lambda: type("C", (), {"bucket": lambda self, name: Bucket()})(),
        )
        assert gs.exists("gs://b/o") is True
        assert seen["retry"] is not None, "exists() must pass a retry policy"

    def test_read_file_bytes_passes_a_retry(self, monkeypatch):
        from orient_express.utils import gs

        seen = {}

        class Blob:
            def download_as_bytes(self, retry=None):
                seen["retry"] = retry
                return b"payload"

        class Bucket:
            def blob(self, path):
                return Blob()

        monkeypatch.setattr(
            gs.storage,
            "Client",
            lambda: type("C", (), {"bucket": lambda self, name: Bucket()})(),
        )
        assert gs.read_file_bytes("gs://b/o") == b"payload"
        assert seen["retry"] is not None, "read_file_bytes() must pass a retry policy"

    def test_house_decorator_logs_each_retried_attempt(self, caplog):
        """Same wording as the policies above, so one grep finds every retry."""
        attempts = []

        @retry(retries=3, initial_timeout=0.001)
        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("boom")
            return "ok"

        with caplog.at_level(logging.WARNING):
            assert flaky() == "ok"

        retried = [r for r in caplog.records if "retrying" in r.getMessage()]
        assert len(retried) == 2, "one warning per retried attempt"
        assert "flaky" in retried[0].getMessage()


class TestPoliciesAreBuiltInOnePlace:
    """Every retry object in the package comes out of `utils.retry`.

    They used to be built per module, and they drifted: the GCS policy in
    `utils.gs` was hand-rolled and retried on a NARROWER set of errors than the
    identical-in-intent policy in `vertex` -- it did not cover `ConnectionError`
    or a read timeout, which is most of how a GCS transfer actually fails. Two of
    the four spellings also logged nothing.
    """

    def test_no_module_builds_its_own_retry(self):
        """The factories are the only construction site for a Retry.

        Parsed rather than grepped: `vertex` mentions DEFAULT_RETRY in a comment
        explaining why the chunked download needs wrapping, and a text match
        reports that as a violation.
        """
        import ast
        import pathlib

        import orient_express

        package = pathlib.Path(orient_express.__file__).parent
        offenders = []
        for path in package.rglob("*.py"):
            if path.name == "retry.py" and path.parent.name == "utils":
                continue  # the one module allowed to construct policies
            for node in ast.walk(ast.parse(path.read_text())):
                built = (
                    isinstance(node, ast.Call)
                    and isinstance(getattr(node.func, "attr", node.func), str)
                    and getattr(node.func, "attr", getattr(node.func, "id", ""))
                    == "Retry"
                )
                imported = isinstance(node, ast.alias) and node.name in (
                    "DEFAULT_RETRY",
                    "Retry",
                )
                if built or imported:
                    offenders.append(
                        f"{path.relative_to(package)}: {ast.dump(node)[:60]}"
                    )
        assert not offenders, (
            f"retry policies built outside utils/retry.py: {offenders}"
        )

    def test_every_gcs_caller_gets_the_same_predicate(self):
        """gs, vertex and the TRT cache syncer must agree on what is transient."""
        from orient_express.predictors.runtime import _TrtCacheGcsSync

        syncer = _TrtCacheGcsSync.__new__(_TrtCacheGcsSync)
        syncer._timeout = 30.0
        policies = [
            GCS_RETRY,
            get_gcs_retry_policy(),
            syncer._bounded_retry_policy(),
        ]

        # The errors a GCS transfer really fails on, which the old hand-rolled
        # policy in utils.gs did not retry.
        for exc in (
            ConnectionError("dropped"),
            api_exceptions.ServiceUnavailable("503"),
        ):
            assert all(p._predicate(exc) for p in policies), f"{exc!r} must be retried"
        for exc in (
            api_exceptions.NotFound("gone"),
            api_exceptions.Forbidden("denied"),
        ):
            assert not any(p._predicate(exc) for p in policies), (
                f"{exc!r} must fail fast"
            )

    def test_bounded_policies_keep_googles_timing_and_only_change_the_budget(self):
        from google.cloud.storage.retry import DEFAULT_RETRY

        bounded = get_gcs_retry_policy(timeout=15.0)
        assert bounded._timeout == 15.0, "the caller's deadline must bound the policy"
        assert (bounded._initial, bounded._maximum, bounded._multiplier) == (
            DEFAULT_RETRY._initial,
            DEFAULT_RETRY._maximum,
            DEFAULT_RETRY._multiplier,
        )
        assert get_gcs_retry_policy()._timeout == DEFAULT_RETRY._timeout

    def test_vertex_factory_is_configurable_but_defaults_to_the_constant(self):
        assert get_vertex_retry_policy()._timeout == VERTEX_RETRY._timeout
        assert get_vertex_retry_policy(timeout=5.0)._timeout == 5.0


class TestUrlLoaderRetriesLikeTheRest:
    """The loader's own retry loop still honours the shared contract.

    UrlImageLoader hand-rolls it because aiohttp is async and api_core is not,
    so what it shares with the policies -- a predicate, backoff, a logged
    attempt -- is asserted here rather than assumed.
    """

    def test_transient_fetch_is_retried_and_logged(self, caplog):
        import asyncio

        from orient_express.predictors import loader as loader_module

        attempts = []

        async def flaky():
            for attempt in range(3):
                try:
                    attempts.append(1)
                    if len(attempts) < 3:
                        raise TimeoutError("slow")
                    return b"payload"
                except Exception as e:  # noqa: BLE001
                    if attempt == 2 or not loader_module._is_transient_fetch_error(e):
                        raise
                    loader_module._log_fetch_retry(e)

        with caplog.at_level(logging.WARNING):
            assert asyncio.run(flaky()) == b"payload"

        retried = [r for r in caplog.records if "retrying" in r.getMessage()]
        assert len(retried) == 2
        assert "http" in retried[0].getMessage()

    def test_permanent_status_is_not_transient(self):
        import aiohttp

        from orient_express.predictors.loader import _is_transient_fetch_error

        def response_error(status):
            return aiohttp.ClientResponseError(None, (), status=status)

        assert _is_transient_fetch_error(response_error(503))
        assert not _is_transient_fetch_error(response_error(404))
