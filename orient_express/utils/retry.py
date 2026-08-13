import functools
import logging
import time


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
    backoff budget; for Google API calls prefer `google.api_core.retry.Retry`,
    which retries on a predicate (see `VERTEX_RETRY` in vertex.py).

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
                    logging.warning(
                        f"{old_func.__name__} failed (attempt {i + 1}/"
                        f"{retry_conf.retries}): {e!r}",
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
