"""Fault injection for the Vertex control plane -- 12 artificial failures.

The Vertex policy is applied by OUR code (`VERTEX_RETRY(fn)(...)`), so the fault
goes straight into the wrapped callable: raise, count the attempts, assert the
policy re-attempted or did not. No network, no project, no credentials.

Run standalone:  uv run python tests/test_retry_faults_vertex.py
Or with the suite: uv run pytest tests/test_retry_faults_vertex.py -v
"""

import logging

import pytest
from google.api_core import exceptions as api_exceptions
from google.api_core.exceptions import RetryError

from orient_express import vertex
from orient_express.utils.retry import get_vertex_retry_policy
from orient_express.vertex import VERTEX_RETRY


def flaky(error, fail_times: int, result="ok"):
    """A callable that raises `error` the first `fail_times` calls."""
    attempts = []

    def call(*args, **kwargs):
        attempts.append(1)
        if len(attempts) <= fail_times:
            raise error
        return result

    return call, attempts


# --------------------------------------------------------------- transient


def test_service_unavailable_is_retried_then_succeeds():
    call, attempts = flaky(api_exceptions.ServiceUnavailable("503"), fail_times=2)
    assert VERTEX_RETRY(call)() == "ok"
    assert len(attempts) == 3, "two failures plus the successful attempt"


def test_deadline_exceeded_is_retried():
    call, attempts = flaky(api_exceptions.DeadlineExceeded("504"), fail_times=1)
    assert VERTEX_RETRY(call)() == "ok"
    assert len(attempts) == 2


def test_internal_server_error_is_retried():
    call, attempts = flaky(api_exceptions.InternalServerError("500"), fail_times=1)
    assert VERTEX_RETRY(call)() == "ok"
    assert len(attempts) == 2


def test_too_many_requests_is_retried():
    """429 is the one a busy pipeline actually hits."""
    call, attempts = flaky(api_exceptions.TooManyRequests("429"), fail_times=1)
    assert VERTEX_RETRY(call)() == "ok"
    assert len(attempts) == 2


def test_aborted_is_retried():
    call, attempts = flaky(api_exceptions.Aborted("409"), fail_times=1)
    assert VERTEX_RETRY(call)() == "ok"
    assert len(attempts) == 2


# --------------------------------------------------------------- permanent


def test_not_found_is_not_retried():
    """The point of a predicate: an absent model must not spend the budget."""
    call, attempts = flaky(api_exceptions.NotFound("no such model"), fail_times=99)
    with pytest.raises(api_exceptions.NotFound):
        VERTEX_RETRY(call)()
    assert len(attempts) == 1


def test_permission_denied_is_not_retried():
    call, attempts = flaky(api_exceptions.PermissionDenied("403"), fail_times=99)
    with pytest.raises(api_exceptions.PermissionDenied):
        VERTEX_RETRY(call)()
    assert len(attempts) == 1


def test_non_google_exception_is_not_retried():
    """A bug in our own argument handling should fail on the first attempt."""
    call, attempts = flaky(ValueError("bad argument"), fail_times=99)
    with pytest.raises(ValueError):
        VERTEX_RETRY(call)()
    assert len(attempts) == 1


def test_original_exception_type_reaches_the_caller():
    """No type laundering: get_vertex_model branches on the real type."""
    call, _ = flaky(api_exceptions.NotFound("gone"), fail_times=99)
    with pytest.raises(api_exceptions.NotFound, match="gone"):
        VERTEX_RETRY(call)()


# --------------------------------------------------------------- budget


def test_sustained_outage_gives_up_with_retry_error():
    """A total outage must end the call, not hang until the pipeline dies.

    A 0.1s budget so the test does not actually wait out the real 60s one.
    """
    policy = get_vertex_retry_policy(timeout=0.1)
    call, attempts = flaky(api_exceptions.ServiceUnavailable("503"), fail_times=999)
    with pytest.raises(RetryError):
        policy(call)()
    assert len(attempts) >= 1


# --------------------------------------------------------------- logging


def test_retried_attempts_are_logged_and_clean_calls_are_not(caplog):
    call, _ = flaky(api_exceptions.ServiceUnavailable("503"), fail_times=2)
    with caplog.at_level(logging.WARNING):
        assert VERTEX_RETRY(call)() == "ok"
    retried = [r for r in caplog.records if "retrying" in r.getMessage()]
    assert len(retried) == 2
    assert "vertex" in retried[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert VERTEX_RETRY(lambda: "ok")() == "ok"
    assert not [r for r in caplog.records if "retrying" in r.getMessage()]


# --------------------------------------------------------------- call site


def test_get_vertex_model_call_site_is_covered(monkeypatch):
    """End to end through a real entry point, not just the bare policy.

    This is the call the issue was filed for: a 503 on Model.list used to abort
    the caller outright.
    """
    attempts = []

    def list_models(*args, **kwargs):
        attempts.append(1)
        if len(attempts) <= 2:
            raise api_exceptions.ServiceUnavailable("503 unavailable")
        return []  # no models -> returns None, no VertexModel to build

    monkeypatch.setattr(vertex, "_last_vertex_init", None)
    monkeypatch.setattr(vertex.aiplatform, "init", lambda **kwargs: None)
    monkeypatch.setattr(vertex.aiplatform.Model, "list", staticmethod(list_models))

    result = vertex.get_vertex_model(
        "my-model", "my-project", "us-central1", raise_exception=False
    )
    assert result is None
    assert len(attempts) == 3, "Model.list must be re-attempted after a 503"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header"]))
