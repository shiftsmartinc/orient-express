"""Ordered, bounded, threaded stage glue for chaining inference streams.

These compose with predict_stream to build multi-stage pipelines (see the
README's POG example): every stage is an iterable transform, results stay
in input order, and a bounded window caps memory. Benchmarked on the POG
chain (download -> detect -> crop -> embed -> vector search -> annotate):
5.2x over the serial per-photo loop.
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
    with ThreadPoolExecutor(max_workers=workers) as pool:
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


def flat_map_stream(fn, source, *, workers: int = 1, prefetch: int = 4):
    """map_stream where fn returns an iterable of outputs (one item -> many).

    The stage that turns one detected image into several crop batches.
    """
    for outputs in map_stream(
        lambda item: list(fn(item)), source, workers=workers, prefetch=prefetch
    ):
        yield from outputs
