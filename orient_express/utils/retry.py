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
    """
    Retry decorated function
    :param retries: the number of retries
    :param initial_timeout: initial timeout before exponential backoff
    :param max_timeout: maximum timeout allowed
    """

    def decorator(old_func):
        retry_conf = RetryConfig(retries, initial_timeout, max_timeout)

        def new_func(*args, **kwargs):
            timeout = retry_conf.initial_timeout
            for i in range(retry_conf.retries):
                try:
                    return old_func(*args, **kwargs)
                except Exception as e:
                    logging.exception(
                        f"Function {old_func.__name__} failed with error {e}, retry count {i}"
                    )
                    time.sleep(timeout)
                    timeout *= 2
                    timeout = min(timeout, retry_conf.max_timeout)
            raise Exception(
                f"Function {old_func.__name__} failed after {retry_conf.retries} retries"
            )

        return new_func

    return decorator
