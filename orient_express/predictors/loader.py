"""Image loading that composes with predict_stream.

Two loaders, chosen by one question — do you have custom loading logic,
or just URLs?

- ImageLoader: runs YOUR `load` callable on threads. Right for local
  files, video frames, crops, custom auth — any source you control.
- UrlImageLoader: items map to URLs of encoded images; the loader owns
  downloading (asyncio, hundreds of concurrent requests) and decoding
  (cv2, GIL-free). Right for the standard case of photos in GCS/HTTP,
  and much faster there.
"""

import asyncio
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from threading import Event, Thread


def _log_load_error(item, exc):
    logging.warning(f"ImageLoader: load failed for {item!r}: {exc}")


class _FetchError:
    """Carries a fatal exception from UrlImageLoader's loop thread."""

    def __init__(self, exc: BaseException):
        self.exc = exc


_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _is_transient_fetch_error(exc: BaseException) -> bool:
    import aiohttp

    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in _RETRYABLE_STATUSES
    return isinstance(
        exc,
        (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, TimeoutError),
    )


def _group_batches(results, batch_size: int):
    """Group non-None results into batch_size lists (None = skipped item)."""
    batch = []
    for result in results:
        if result is None:
            continue
        batch.append(result)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
        workers: int = 32,
        prefetch: int = 2,
        keep_original: bool = False,
        on_error=None,
    ):
        # workers=32 default: per-object GCS latency is ~0.5s while the
        # transfer itself takes ~ms, so throughput ≈ workers/latency
        # until CPU binds — a small pool (8 workers ≈ 16 img/s) caps the
        # pipeline well below what a typical machine can decode
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
        # not a `with` block: on early consumer exit (break/GC closes the
        # generator at the yield) Executor.__exit__ would run every queued
        # load before returning — with a hanging `load` callable, forever.
        # cancel_futures drops the queued window; close() only waits out
        # the loads already executing.
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
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
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

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


class UrlImageLoader:
    """Download and decode images from URLs — the high-throughput loader.

    Where ImageLoader runs your `load` callable on worker threads, this
    loader owns the whole fetch+decode path for the common case "my items
    are URLs of encoded images". Downloads run on an asyncio event loop
    (object-store latency demands ~rate x latency requests in flight,
    which thread pools pay GIL tax to hold; one event loop holds hundreds
    of connections for free), and decoding runs on a small thread pool
    via decode_image (cv2 releases the GIL).

        loader = UrlImageLoader(rows, url=lambda r: r["image_url"])
        for rows_batch, preds in predictor.predict_stream(loader, confidence=0.4):
            ...

    URLs are fetched exactly as given — no credentials are attached. Pass
    `headers` when the endpoint needs auth.

    decode="exact" (default) is pixel-faithful to a full decode.
    decode="fast" decodes JPEGs at a reduced scale sized to the target
    resolution — faster, but pixels differ: validate model accuracy
    before using it in production. Prediction coordinates stay in
    original-photo pixels, UNLESS keep_original=True, in which case they
    match the reduced image in the payload (so cropping it works
    unchanged) — see iter_feeds for the full contract.

    Transient fetch errors (HTTP 408/429/5xx, connection drops, timeouts)
    are retried `retries` more times with exponential backoff (retry_backoff
    seconds, doubling per attempt) — object stores throw occasional 503s at
    this concurrency, and without retries each one would silently drop an
    item. Failed downloads (after retries) and undecodable files are
    skipped and reported to `on_error`, and results keep input order — the
    same contract as ImageLoader, so predict_stream fuses with either
    interchangeably.
    """

    def __init__(
        self,
        items,
        url=None,
        *,
        batch_size: int = 16,
        concurrency: int = 128,
        decode: str = "exact",
        fast_size: tuple[int, int] | None = None,
        decode_threads: int = 4,
        prefetch: int = 2,
        keep_original: bool = False,
        on_error=None,
        headers: dict | None = None,
        timeout: float = 60.0,
        retries: int = 2,
        retry_backoff: float = 0.5,
    ):
        if decode not in ("exact", "fast"):
            raise ValueError(f"decode must be 'exact' or 'fast', got {decode!r}")
        self.items = items
        self.url = url if url is not None else lambda item: item
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.decode = decode
        self.fast_size = fast_size
        self.decode_threads = decode_threads
        self.prefetch = prefetch
        self.keep_original = keep_original
        self.on_error = on_error or _log_load_error
        self.headers = headers or {}
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff

    # ------------------------------------------------------------ fetch

    def _byte_stream(self, window_size: int):
        """Yield (item, bytes) in input order from a background event loop.

        Failed fetches are reported to on_error and yielded as
        (item, None). The loop thread is a daemon and aborts promptly if
        the consumer stops early (stop flag + queue drain, mirroring
        predict_stream's producer hygiene).
        """
        out: Queue = Queue(maxsize=window_size)
        stop = Event()
        sentinel = object()

        def put(value) -> bool:
            while not stop.is_set():
                try:
                    out.put(value, timeout=0.2)
                    return True
                except Full:
                    continue
            return False

        async def main():
            import aiohttp

            loop = asyncio.get_running_loop()
            connector = aiohttp.TCPConnector(limit=self.concurrency)
            client_timeout = aiohttp.ClientTimeout(total=self.timeout)

            async with aiohttp.ClientSession(
                connector=connector, timeout=client_timeout
            ) as session:

                async def fetch(item):
                    url = self.url(item)
                    for attempt in range(self.retries + 1):
                        try:
                            async with session.get(
                                url, headers=self.headers
                            ) as response:
                                response.raise_for_status()
                                return await response.read()
                        except Exception as e:  # noqa: BLE001 - filtered below
                            if attempt == self.retries or not _is_transient_fetch_error(
                                e
                            ):
                                raise
                        await asyncio.sleep(self.retry_backoff * 2**attempt)

                pending: deque = deque()
                iterator = iter(self.items)

                def top_up():
                    while len(pending) < self.concurrency and not stop.is_set():
                        try:
                            item = next(iterator)
                        except StopIteration:
                            return
                        pending.append((item, asyncio.create_task(fetch(item))))

                top_up()
                while pending and not stop.is_set():
                    item, task = pending.popleft()
                    try:
                        data = await task
                    except Exception as e:  # noqa: BLE001 - per-item tolerance
                        self.on_error(item, e)
                        data = None
                    if not await loop.run_in_executor(None, put, (item, data)):
                        break
                    top_up()
                for _, task in pending:
                    task.cancel()

        def run():
            try:
                asyncio.run(main())
            except Exception as e:  # noqa: BLE001 - surfaced via queue
                put(_FetchError(e))
            finally:
                # the sentinel must reach the consumer on EVERY exit path:
                # a BaseException here (e.g. CancelledError, which is not an
                # Exception) would otherwise leave it parked in out.get()
                # forever, silently freezing the whole pipeline
                put(sentinel)

        Thread(target=run, daemon=True, name="url-image-loader").start()
        try:
            while True:
                value = out.get()
                if value is sentinel:
                    return
                if isinstance(value, _FetchError):
                    raise value.exc
                yield value
        finally:
            stop.set()
            while True:
                try:
                    out.get_nowait()
                except Empty:
                    break

    # ------------------------------------------------------------ decode

    def _decoded_stream(self, per_item, prefetch: int):
        """Apply per_item(item, data) on decode threads, in order.

        The fetch side looks ahead up to `concurrency` raw byte payloads
        (a few hundred KB each); the decode side keeps only a small
        window of decoded results in flight — decoded images are the
        memory-heavy stage, and decode is fast enough not to need depth.
        """

        def safe(pair):
            item, data = pair
            if data is None:
                return None
            try:
                return per_item(item, data)
            except Exception as e:  # noqa: BLE001 - per-item tolerance
                self.on_error(item, e)
                return None

        decode_window = max(self.batch_size * (prefetch + 1), self.decode_threads * 2)
        # try/finally instead of `with`: cancel queued decodes on early
        # consumer exit rather than running the whole window (see _stream)
        pool = ThreadPoolExecutor(max_workers=self.decode_threads)
        try:
            pending: deque = deque()
            for pair in self._byte_stream(self.concurrency):
                pending.append(pool.submit(safe, pair))
                while len(pending) >= decode_window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _fast_target(self, default):
        if self.decode != "fast":
            return None
        target = self.fast_size or default
        if target is None:
            raise ValueError(
                "decode='fast' needs a target size: pass fast_size=(w, h), "
                "or iterate via predict_stream, which supplies the model's "
                "input resolution."
            )
        return target

    def __iter__(self):
        from ..utils.image_processor import decode_image

        target = self._fast_target(None) if self.decode == "fast" else None

        def to_image(item, data):
            return item, decode_image(data, fast_target=target)

        for batch in _group_batches(
            self._decoded_stream(to_image, self.prefetch), self.batch_size
        ):
            items = [item for item, _ in batch]
            payload = batch if self.keep_original else items
            yield payload, [image for _, image in batch]

    def iter_feeds(self, predictor, prefetch: int | None = None):
        """Fused iteration for predict_stream: yields (payload, feed).

        Decode + the predictor's per-item resize run on the decode
        threads; with decode="fast" the JPEG reduction is sized to the
        predictor's input resolution automatically.

        Coordinate space under decode="fast" (with "exact" the two cases
        coincide, since the decoded image IS the original):

        - keep_original=False: the model is told the ORIGINAL photo's
          upright dimensions, so predictions (boxes, masks) come back in
          original-photo pixels — the same values decode="exact" yields.
        - keep_original=True: the model is told the decoded (reduced)
          image's dimensions, so predictions line up with the image in the
          payload — cropping detections out of it works unchanged. To map
          such predictions onto the original file, scale by
          original_width / decoded_width.
        """
        from ..utils.image_processor import decode_image

        if prefetch is None:
            prefetch = self.prefetch
        target = (
            self._fast_target(predictor.img_size) if self.decode == "fast" else None
        )

        def to_feed_part(item, data):
            if target is not None and not self.keep_original:
                image, original = decode_image(
                    data, fast_target=target, return_original_size=True
                )
                array, _ = predictor.preprocess_item(image)
                size = (original[1], original[0])  # (height, width) feed order
            else:
                image = decode_image(data, fast_target=target)
                array, size = predictor.preprocess_item(image)
            return item, (image if self.keep_original else None), array, size

        for batch in _group_batches(
            self._decoded_stream(to_feed_part, prefetch), self.batch_size
        ):
            items = [b[0] for b in batch]
            payload = [(b[0], b[1]) for b in batch] if self.keep_original else items
            feed = predictor.assemble_feed([b[2] for b in batch], [b[3] for b in batch])
            yield payload, feed
