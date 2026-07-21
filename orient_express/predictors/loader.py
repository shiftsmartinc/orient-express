"""Generic threaded image loading that composes with predict_stream."""

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def _log_load_error(item, exc):
    logging.warning(f"ImageLoader: load failed for {item!r}: {exc}")


class ImageLoader:
    """Turn an iterable of items into image batches, loading on threads.

    `load` is any per-item callable returning a PIL image — download a URL,
    read a file, decode a video frame, crop a larger image. Loads run on
    `workers` threads with a bounded look-ahead window, so iterating this
    from predict_stream overlaps loading with GPU inference and never holds
    more than roughly batch_size * (prefetch + 1) images in memory:

        loader = ImageLoader(rows, load=lambda row: download(row["image_url"]),
                             batch_size=32, workers=8)
        for rows_batch, preds in predictor.predict_stream(loader, confidence=0.4):
            for row, pred in zip(rows_batch, preds):
                ...

    predict_stream fuses with this loader: the predictor's per-item resize
    runs inside the worker that loaded the image (no second preprocess pool,
    no queue of full-size images). Set keep_original=True when downstream
    stages need the original images (e.g. cropping detections) — the batch
    payload then holds (item, image) pairs instead of items.

    Items whose load or per-item preprocessing raises are skipped and
    reported to `on_error` (default: log a warning), so one bad input
    doesn't kill the stream; pass a collecting callback to record failures.
    PIL decodes lazily, so a corrupt or truncated file often raises only
    when the pixels are first touched — that decode is forced inside the
    same guarded worker task, so such items are skipped too.

    Iterating this directly yields (payload, images) batches — the same
    shape any hand-written source produces, so anything predict_stream
    accepts, this can be swapped for.
    """

    def __init__(
        self,
        items,
        load,
        *,
        batch_size: int = 16,
        workers: int = 8,
        prefetch: int = 2,
        keep_original: bool = False,
        on_error=None,
    ):
        self.items = items
        self.load = load
        self.batch_size = batch_size
        self.workers = workers
        self.prefetch = prefetch
        self.keep_original = keep_original
        self.on_error = on_error or _log_load_error

    def _stream(self, work, prefetch: int | None = None):
        """Run work(item) on the pool inside a bounded, ordered window."""
        if prefetch is None:
            prefetch = self.prefetch
        window_size = max(self.batch_size * (prefetch + 1), self.workers)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            window: deque = deque()
            it = iter(self.items)

            def fill():
                while len(window) < window_size:
                    try:
                        window.append(pool.submit(work, next(it)))
                    except StopIteration:
                        return

            fill()
            while window:
                out = window.popleft().result()
                fill()
                yield out

    def _batches(self, results):
        """Group non-failed results into batch_size lists."""
        batch = []
        for result in results:
            if result is None:  # failed load, already reported
                continue
            batch.append(result)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __iter__(self):
        def safe_load(item):
            try:
                image = self.load(item)
                image.load()  # force PIL's lazy decode while still guarded
                return item, image
            except Exception as e:  # noqa: BLE001 - per-item fault tolerance
                self.on_error(item, e)
                return None

        for batch in self._batches(self._stream(safe_load)):
            items = [item for item, _ in batch]
            images = [image for _, image in batch]
            payload = batch if self.keep_original else items
            yield payload, images

    def iter_feeds(self, predictor, prefetch: int | None = None):
        """Fused iteration for predict_stream: yields (payload, feed).

        The predictor's preprocess_item (resize + size capture) runs in the
        same worker task as the load; assemble_feed turns each batch into
        the exact feed preprocess() would have produced. `prefetch`
        overrides the loader's own look-ahead when given.
        """

        def load_and_preprocess(item):
            try:
                image = self.load(item)
                array, size = predictor.preprocess_item(image)
            except Exception as e:  # noqa: BLE001 - per-item fault tolerance
                self.on_error(item, e)
                return None
            return item, (image if self.keep_original else None), array, size

        for batch in self._batches(self._stream(load_and_preprocess, prefetch)):
            items = [b[0] for b in batch]
            payload = [(b[0], b[1]) for b in batch] if self.keep_original else items
            feed = predictor.assemble_feed([b[2] for b in batch], [b[3] for b in batch])
            yield payload, feed
