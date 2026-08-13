"""Every retry policy in the package, and the functions that build them.

One home for two reasons. Policies drifted when each module built its own: the
GCS calls in `utils.gs` were retrying on a narrower error set than the ones in
`vertex`, missing exactly the `ConnectionError` and `requests.Timeout` that GCS
transfers actually raise, and two of the four spellings of "the GCS policy"
logged nothing. And the choice of mechanism is not obvious per call site, so it
is made once here:

* `get_vertex_retry_policy` / `get_gcs_retry_policy` -- `google.api_core`
  policies for Google API calls. They retry on a PREDICATE, so a missing model
  or a permission error fails immediately instead of burning the backoff
  budget, and the original exception reaches the caller.
* `retry` -- the house decorator, for everything else (an image over plain
  HTTP). Best-effort: it retries on any exception.

Callers do not construct `Retry` themselves. Modules that always want the same
policy hold it as a constant (`vertex.VERTEX_RETRY`); modules whose budget
depends on runtime state call the function with a `timeout`.
"""

from __future__ import annotations

import functools
import logging
import time

from google.api_core import exceptions as api_exceptions
from google.api_core import retry as api_retry
from google.api_core.retry import Retry

# Transient gRPC statuses for the Vertex control plane. Everything absent from
# this list -- 404, 403, malformed request -- is permanent and must fail on the
# first attempt.
VERTEX_TRANSIENT_ERRORS = (
    api_exceptions.ServiceUnavailable,  # 503
    api_exceptions.DeadlineExceeded,
    api_exceptions.InternalServerError,
    api_exceptions.Aborted,
    api_exceptions.TooManyRequests,  # 429
)

# Total budget across attempts for the Vertex control plane, seconds: a sustained
# outage fails the caller in about a minute rather than hanging until the
# pipeline's own timeout.
VERTEX_TIMEOUT_SECONDS = 60.0


def get_retry_logger(transport: str):
    """An `on_error` callback that records each retried attempt.

    A silent retry is nearly as bad as no retry: the call succeeds, nobody learns
    the dependency wobbled, and the next incident looks like the first. WARNING
    rather than INFO because a remote call already failed once to get here.

    Use where the policy is built here and `on_error` can be passed. For a copy
    of somebody else's policy use `get_logged_retry_policy` instead.
    """

    def on_error(exc: BaseException) -> None:
        logging.warning("%s call failed, retrying: %r", transport, exc)

    return on_error


def get_logged_retry_policy(policy: Retry, transport: str) -> Retry:
    """A copy of `policy` that logs each retriable failure as it is decided.

    For policies we did not build -- `google.cloud.storage.retry.DEFAULT_RETRY`
    above all. `api_core` exposes no `with_on_error`, so the callback cannot be
    attached to a copy; wrapping the predicate is the only hook that does not mean
    rebuilding the policy and duplicating Google's list of retriable errors, which
    would then silently drift from the library.

    Timing and the retriable-error set are Google's, unchanged -- only the logging
    is added. Permanent failures are not logged: those are the caller's to report.
    """
    inner = policy._predicate

    def predicate(exc: BaseException) -> bool:
        retriable = inner(exc)
        if retriable:
            logging.warning("%s call failed, retrying: %r", transport, exc)
        return retriable

    return policy.with_predicate(predicate)


def get_vertex_retry_policy(timeout: float = VERTEX_TIMEOUT_SECONDS) -> Retry:
    """The policy for Vertex AI control-plane calls (gRPC/GAPIC).

    Applied to every remote call in `vertex` rather than at whichever site looked
    risky: ad-hoc coverage is what let a transient 503 on `Model.list` fail a
    whole pipeline run.

    :param timeout: total budget across all attempts, seconds
    """
    return api_retry.Retry(
        predicate=api_retry.if_exception_type(*VERTEX_TRANSIENT_ERRORS),
        initial=1.0,
        multiplier=2.0,
        maximum=10.0,
        timeout=timeout,
        on_error=get_retry_logger("vertex"),
    )


def get_gcs_retry_policy(timeout: float | None = None) -> Retry:
    """The policy for Google Cloud Storage calls.

    Google's own `DEFAULT_RETRY`, with logging added and nothing else changed.
    Reusing it rather than hand-rolling a policy matters more here than for
    Vertex: the storage predicate covers the transport's real failure modes --
    a dropped connection, a read timeout, a truncated chunked response -- which
    a list of `api_core` status exceptions does not.

    :param timeout: total budget across attempts, seconds; None keeps Google's
        (120s). Pass one wherever the caller has a deadline of its own, so the
        policy cannot keep retrying past it.
    """
    # Imported here, not at module scope: this module is imported by the
    # inference servers and by `predictors.runtime`, which both keep
    # `google.cloud.storage` off the import path until something needs it.
    from google.cloud.storage.retry import DEFAULT_RETRY

    policy = DEFAULT_RETRY if timeout is None else DEFAULT_RETRY.with_timeout(timeout)
    return get_logged_retry_policy(policy, "gcs")


class RetryConfig:
    def __init__(self, retries, initial_timeout, max_timeout):
        self.retries = retries
        self.initial_timeout = initial_timeout
        self.max_timeout = max_timeout


def retry(
    retries: int = 3,
    initial_timeout: float = 0.5,
    max_timeout: float = 30,
):
    """Retry decorated function on any exception, with exponential backoff.

    Best-effort by design: it retries *everything*, so it suits work where any
    failure is worth another attempt and the caller only needs the result --
    fetching an image, for example. It is the wrong tool where the caller must
    distinguish error kinds, or where a permanent failure should not spend the
    backoff budget; for Google API calls use `get_vertex_retry_policy` or
    `get_gcs_retry_policy` above, which retry on a predicate.

    The final failure is re-raised as-is, so the original exception type and
    traceback reach the caller.

    :param retries: total number of attempts, not attempts after the first
    :param initial_timeout: first backoff, doubling per attempt
    :param max_timeout: cap on a single backoff
    """

    def decorator(old_func):
        retry_conf = RetryConfig(retries, initial_timeout, max_timeout)

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            timeout = retry_conf.initial_timeout
            for i in range(retry_conf.retries):
                try:
                    return old_func(*args, **kwargs)
                except Exception as e:
                    last_attempt = i == retry_conf.retries - 1
                    # Same wording as the api_core policies above ("... failed,
                    # retrying: ...") so one grep finds every retried attempt
                    # regardless of which mechanism protected the call.
                    logging.warning(
                        f"{old_func.__name__} call failed"
                        f"{'' if last_attempt else ', retrying'} "
                        f"(attempt {i + 1}/{retry_conf.retries}): {e!r}",
                        exc_info=not last_attempt,
                    )
                    # Re-raise the original rather than wrapping it: callers
                    # need the real type to tell a transient failure from a
                    # permanent one, and the traceback points at the real cause.
                    if last_attempt:
                        raise
                    # Only sleep between attempts -- sleeping after the last one
                    # delays the inevitable for no benefit.
                    time.sleep(timeout)
                    timeout = min(timeout * 2, retry_conf.max_timeout)

        return new_func

    return decorator
