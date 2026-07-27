"""Ordered, bounded, threaded stage glue for chaining inference streams.

These compose with predict_stream to build multi-stage pipelines (see the
README's detect -> crop -> embed -> search -> annotate example): every
stage is an iterable transform, results stay in input order, and a bounded
window caps memory, so all stages run concurrently instead of one photo at
a time.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def map_stream(fn, source, *, workers: int = 1, prefetch: int = 4):
    """Apply fn to each item of source on worker threads, yielding in order.

    The window of in-flight work is bounded by max(prefetch, workers), so a
    slow consumer never causes unbounded queueing. An fn that raises stops
    the stream at that item's position (wrap fn for per-item tolerance).
    """
    it = iter(source)
    # not a `with` block: on early consumer exit (break/GC closes the
    # generator here at the yield) Executor.__exit__ would run every queued
    # future before returning; cancel_futures drops them so close() only
    # waits out the tasks already executing
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        window: deque = deque()

        def fill():
            while len(window) < max(prefetch, workers):
                try:
                    window.append(pool.submit(fn, next(it)))
                except StopIteration:
                    return

        fill()
        while window:
            out = window.popleft().result()
            fill()
            yield out
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def flat_map_stream(fn, source, *, workers: int = 1, prefetch: int = 4):
    """map_stream where fn returns an iterable of outputs (one item -> many).

    The stage that turns one detected image into several crop batches.
    """
    for outputs in map_stream(
        lambda item: list(fn(item)), source, workers=workers, prefetch=prefetch
    ):
        yield from outputs
