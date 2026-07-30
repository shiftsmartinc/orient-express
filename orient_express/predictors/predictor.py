import os
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np
import yaml
from PIL import Image

from ..utils.colors import generate_color_scheme
from ..utils.image_processor import image_to_array, image_to_base64
from ..utils.paths import get_metadata_path
from .runtime import _TRT_DEVICES, Device, create_session, parse_trt_profile_shapes

IMAGE_ONNX_IMAGE_REPO = (
    "us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx"
)

# Threshold for threading collate_images (total input pixels across the
# batch); below it, per-task dispatch overhead outweighs the resize work.
THREADED_COLLATE_MIN_TOTAL_PIXELS = 8_000_000


def get_image_onnx_container_uri() -> str:
    """Serving-image URI whose tag tracks the installed library version.

    The Makefile builds/pushes the image with the same version tag, so the
    library and its serving image can't drift apart.
    """
    try:
        tag = f"v{_package_version('orient_express')}"
    except PackageNotFoundError:  # running from a source tree without install
        tag = "latest"
    return f"{IMAGE_ONNX_IMAGE_REPO}:{tag}"


# model_type string (persisted in every uploaded metadata.yaml) -> class.
# Populated automatically when a Predictor subclass defines `model_type`.
PREDICTOR_REGISTRY: dict[str, type["Predictor"]] = {}


class _ProducerError:
    """Carries an exception from predict_stream's producer thread."""

    def __init__(self, exc: BaseException):
        self.exc = exc


class Predictor(ABC):
    model_type: str
    model_path: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        model_type = cls.__dict__.get("model_type")
        if isinstance(model_type, str):
            existing = PREDICTOR_REGISTRY.get(model_type)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"model_type '{model_type}' is already registered by "
                    f"{existing.__name__}; model_type strings must be unique "
                    "(they are persisted in uploaded model metadata)"
                )
            PREDICTOR_REGISTRY[model_type] = cls

    @classmethod
    def from_dir(
        cls, dir: str, metadata: dict, device: str = Device.CPU, **kwargs
    ) -> "Predictor":
        """Construct this predictor from a downloaded artifact directory.

        Extra keyword arguments are forwarded to the constructor (e.g.
        provider_options for ImagePredictors).
        """
        raise NotImplementedError(f"{cls.__name__} does not implement from_dir")

    @abstractmethod
    def get_serving_container_image_uri(self) -> str:
        pass

    @abstractmethod
    def get_serving_container_health_route(self, model_name) -> str:
        pass

    @abstractmethod
    def get_serving_container_predict_route(self, model_name) -> str:
        pass

    @abstractmethod
    def dump(self, dir: str) -> list[str]:
        pass


class ImagePredictor(Predictor):
    """Image predictor backed directly by an ONNX Runtime session.

    ORT is the only backend: its TensorRT execution provider runs close
    enough to native TensorRT that a second backend was not worth the
    complexity.

    Inference is split into three public stages so callers can pipeline:

        feed = predictor.preprocess(images)     # CPU: collate/resize
        outputs = predictor.infer(feed)         # GPU: session.run
        preds = predictor.postprocess(outputs, feed, **kwargs)  # CPU

    predict() composes the three; predict_stream() overlaps the CPU stages
    with GPU inference across consecutive batches (cv2.resize and
    session.run both release the GIL, so threads give real overlap).
    """

    model_type: str

    def __init__(
        self,
        model_path: str,
        classes: dict[int, str] | None = None,
        device: str = Device.CPU,
        provider_options: dict | None = None,
    ):
        self._trt_profile_bounds = None
        if device in _TRT_DEVICES and provider_options:
            # Validate profile syntax before the session is created: ORT
            # ignores a malformed spec with only a log warning and profiles
            # the first shape it sees instead of the intended range — and a
            # late error would come after the full engine build.
            parsed = {}
            for key in (
                "trt_profile_min_shapes",
                "trt_profile_opt_shapes",
                "trt_profile_max_shapes",
            ):
                spec = provider_options.get(key)
                if spec:
                    parsed[key] = parse_trt_profile_shapes(spec)
            if (
                "trt_profile_min_shapes" in parsed
                and "trt_profile_max_shapes" in parsed
            ):
                self._trt_profile_bounds = (
                    parsed["trt_profile_min_shapes"],
                    parsed["trt_profile_max_shapes"],
                )

        self.session, self._trt_cache_sync = create_session(
            model_path, device, provider_options
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        input_shape = self.session.get_inputs()[0].shape
        # This library's contract is NHWC uint8 image inputs (see
        # MODEL_REQUIREMENTS.md); an NCHW export would silently yield
        # resolution=3 and garbage resizes, so refuse it loudly.
        if len(input_shape) == 4 and input_shape[1] == 3 and input_shape[3] != 3:
            raise ValueError(
                f"Model input '{self.input_names[0]}' has shape {input_shape}, "
                "which looks channels-first (NCHW). This library requires "
                "NHWC image inputs ([batch, height, width, 3]) — re-export "
                "the model with channels-last inputs."
            )
        self.resolution = input_shape[1]
        self.img_size = (self.resolution, self.resolution)

        self.classes = classes or {}
        self.color_scheme = generate_color_scheme(list(self.classes.values()))
        self.model_path = model_path
        self._seen_feed_shapes: set[tuple] = set()

    def preprocess(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        """CPU stage: images -> feed dict.

        Subclasses may add entries that are not model inputs (e.g. semantic
        segmentation's target_sizes); infer() only passes input_names to the
        session, and postprocess() receives the whole feed for such context.
        """
        return {self.input_names[0]: self.collate_images(images)}

    def preprocess_item(self, image: Image.Image):
        """Per-image slice of preprocess, for fused loading.

        ImageLoader runs this inside the worker that loaded the image, so
        resize cost spreads across the loader threads and full-size images
        never queue up. Returns (resized_array, (height, width));
        assemble_feed() turns a batch of these into the same feed
        preprocess() would build.
        """
        return (
            cv2.resize(image_to_array(image), self.img_size),
            (image.size[1], image.size[0]),
        )

    def assemble_feed(self, arrays, sizes) -> dict[str, np.ndarray]:
        """Batch of preprocess_item results -> feed dict.

        Must produce exactly what preprocess() would for the same images;
        subclasses that override preprocess() override this to match (there
        is a test pinning the equivalence per predictor).
        """
        return {self.input_names[0]: np.stack(arrays)}

    def infer(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """GPU stage: pure session.run. Releases the GIL while running."""
        if self._trt_profile_bounds is not None:
            self._check_trt_profile(feed)
        outputs = self.session.run(
            None, {name: feed[name] for name in self.input_names}
        )
        shapes = tuple(feed[name].shape for name in self.input_names)
        if shapes not in self._seen_feed_shapes:
            self._seen_feed_shapes.add(shapes)
            if self._trt_cache_sync is not None:
                # TRT builds engines lazily, and only a run with a first-seen
                # input shape can trigger a build; push fresh cache files to
                # GCS in the background so the next worker skips the
                # multi-minute build
                self._trt_cache_sync.schedule_upload()
        return outputs

    def _check_trt_profile(self, feed):
        """Raise on inputs outside the declared TensorRT optimization profile.

        Handing ORT an out-of-profile input fails the call and silently
        falls back to CUDA for it — in production that is an invisible
        performance downgrade. Profiles are mandatory for TensorRT devices,
        so every input is checked against the declared min/max range.
        """
        lo, hi = self._trt_profile_bounds
        for name in self.input_names:
            shape = tuple(feed[name].shape)
            lo_dims, hi_dims = lo.get(name), hi.get(name)
            if lo_dims is None or hi_dims is None:
                continue
            fits = len(shape) == len(lo_dims) == len(hi_dims) and all(
                lo_d <= dim <= hi_d
                for lo_d, dim, hi_d in zip(lo_dims, shape, hi_dims, strict=True)
            )
            if not fits:
                raise ValueError(
                    f"TensorRT: input '{name}' has shape {shape}, outside "
                    f"the declared optimization profile "
                    f"[{lo_dims}..{hi_dims}]. ORT would fail the call and "
                    "silently fall back to CUDA for it. Widen "
                    "trt_profile_min/max_shapes in provider_options to "
                    "cover this shape."
                )

    def postprocess(self, outputs, feed, **kwargs):
        """CPU stage: raw session outputs -> prediction objects.

        Receives the feed for context (e.g. target sizes). Subclasses define
        their own keyword arguments (e.g. confidence), which predict_stream
        passes through.
        """
        raise NotImplementedError

    def predict_stream(self, batches, *, prefetch: int | None = None, **kwargs):
        """Pipelined predict over any iterable of image batches.

        A batch is either a list of PIL images, or a (payload, images) tuple
        — the payload is opaque and comes back untouched with that batch's
        predictions, so URLs, dataframe rows, file paths or indices ride
        along with their results:

            for rows, preds in predictor.predict_stream(my_batches(), confidence=0.4):
                for row, pred in zip(rows, preds):
                    ...

        Results yield in input order, same values as predict() per batch.
        The source iterable is pulled on a background thread, so a generator
        that blocks on IO (downloads, disk reads) runs concurrently with GPU
        inference. preprocess/postprocess run on worker threads and overlap
        with infer on neighboring batches; prefetch bounds how many batches
        are in flight (memory bound) — default 2, or the loader's own
        prefetch for an ImageLoader source; an explicit value (an int >= 1)
        overrides both. Keyword arguments (e.g. confidence) are passed to
        postprocess.

        An ImageLoader source takes the fused fast path: each image is
        resized by the same worker that loaded it (via preprocess_item), so
        there is no separate preprocess pool and full-size images never
        accumulate — faster than the generic path when loading is CPU-bound,
        identical results.
        """
        if prefetch is not None and not (isinstance(prefetch, int) and prefetch >= 1):
            raise ValueError(f"prefetch must be an int >= 1, got {prefetch!r}")

        # loaders (ImageLoader, UrlImageLoader, or anything implementing
        # iter_feeds) take the fused fast path: per-item preprocess runs
        # inside the loader's own workers
        if hasattr(batches, "iter_feeds"):
            yield from self._predict_stream_fused(batches, prefetch, **kwargs)
            return
        if prefetch is None:
            prefetch = 2

        sentinel = object()
        stop = Event()
        source: Queue = Queue(maxsize=max(1, prefetch))

        def produce():
            try:
                for batch in batches:
                    if stop.is_set():
                        return
                    source.put(batch)
                source.put(sentinel)
            except BaseException as e:  # noqa: BLE001 - reraised on consumer
                source.put(_ProducerError(e))

        Thread(target=produce, daemon=True).start()

        try:
            yield from self._pipeline(source, sentinel, prefetch, kwargs)
        finally:
            # If the consumer stopped early (break, error, GC of the
            # generator), the producer may be parked in source.put() on the
            # full queue; without this it would block forever, leaking the
            # thread and the batches it holds. Draining frees a slot, its
            # put() completes, and the stop flag ends its loop.
            stop.set()
            while True:
                try:
                    source.get_nowait()
                except Empty:
                    break

    def _pipeline(self, source, sentinel, prefetch, kwargs):
        """predict_stream's consumer loop: preprocess/infer/postprocess."""
        no_payload = object()

        def split(batch):
            if isinstance(batch, tuple) and len(batch) == 2:
                return batch
            return no_payload, batch

        workers = max(2, prefetch)
        # Postprocess results are drained opportunistically, so a postprocess
        # stage slower than infer would otherwise queue without bound — and
        # every queued future holds its whole batch's outputs and feed. Cap
        # the queue and block (backpressure on infer) when it fills.
        post_cap = 2 * workers
        # try/finally instead of `with`: on early consumer exit, cancel the
        # queued preprocess/postprocess futures instead of running them all
        # before close() returns
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            pre: deque = deque()  # (payload, n_images, preprocess future)
            post: deque = deque()  # (payload, postprocess future)
            exhausted = False

            def top_up():
                nonlocal exhausted
                while not exhausted and len(pre) < prefetch:
                    batch = source.get()
                    if batch is sentinel:
                        exhausted = True
                        return
                    if isinstance(batch, _ProducerError):
                        exhausted = True
                        raise batch.exc
                    payload, images = split(batch)
                    pre.append(
                        (payload, len(images), pool.submit(self.preprocess, images))
                    )

            def emit(payload, result):
                return result if payload is no_payload else (payload, result)

            top_up()
            while pre or post:
                # yield due results without blocking the infer loop — unless
                # nothing is left to infer, or postprocess has fallen behind
                # (queue at cap)
                while post and (post[0][1].done() or not pre or len(post) >= post_cap):
                    payload, fut = post.popleft()
                    yield emit(payload, fut.result())
                if pre:
                    payload, n_images, fut = pre.popleft()
                    feed = fut.result()
                    top_up()
                    if n_images == 0:
                        post.append((payload, pool.submit(list)))
                        continue
                    outputs = self.infer(feed)
                    post.append(
                        (
                            payload,
                            pool.submit(self.postprocess, outputs, feed, **kwargs),
                        )
                    )
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def _predict_stream_fused(self, loader, prefetch=None, **kwargs):
        """predict_stream fast path for ImageLoader: infer + threaded post."""
        # try/finally instead of `with`: cancel queued postprocess futures
        # on early consumer exit (see _pipeline)
        pool = ThreadPoolExecutor(max_workers=2)
        post_cap = 4  # same bounded-postprocess backpressure as _pipeline
        try:
            post: deque = deque()
            for payload, feed in loader.iter_feeds(self, prefetch):
                while post and (post[0][1].done() or len(post) >= post_cap):
                    done_payload, fut = post.popleft()
                    yield done_payload, fut.result()
                outputs = self.infer(feed)
                post.append(
                    (payload, pool.submit(self.postprocess, outputs, feed, **kwargs))
                )
            while post:
                done_payload, fut = post.popleft()
                yield done_payload, fut.result()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def collate_sizes(self, pil_images: list[Image.Image]):
        sizes = [[img.size[1], img.size[0]] for img in pil_images]
        return np.array(sizes, dtype=np.float32)

    def collate_images(self, pil_images: list[Image.Image]):
        n = len(pil_images)
        batch = np.empty((n, self.resolution, self.resolution, 3), dtype=np.uint8)

        def collate_one(i):
            batch[i] = cv2.resize(image_to_array(pil_images[i]), self.img_size)

        # cv2.resize releases the GIL, so batches of full-size photos collate
        # several times faster on a thread pool; batches of small crops (e.g.
        # from build_vector_index) stay serial — task dispatch would dominate
        # their ~microsecond resizes. The threshold is calibrated empirically.
        total_input_pixels = sum(img.size[0] * img.size[1] for img in pil_images)
        if n >= 2 and total_input_pixels >= THREADED_COLLATE_MIN_TOTAL_PIXELS:
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
                list(pool.map(collate_one, range(n)))
        else:
            for i in range(n):
                collate_one(i)
        return batch

    @classmethod
    def from_dir(cls, dir: str, metadata: dict, device: str = Device.CPU, **kwargs):
        if "model_file" not in metadata:
            raise Exception("No model_file defined in metadata.yaml")
        if "classes" not in metadata:
            raise Exception("No classes defined in metadata.yaml")
        onnx_path = os.path.join(dir, metadata["model_file"])
        return cls(onnx_path, metadata["classes"], device, **kwargs)

    def get_serving_container_image_uri(self):
        return get_image_onnx_container_uri()

    def get_serving_container_health_route(self, model_name):
        return f"/v1/models/{model_name}"

    def get_serving_container_predict_route(self, model_name):
        return f"/v1/models/{model_name}:predict"

    def to_response(self, image: Image.Image, prediction, include_debug: bool = True):
        """Per-image response dict served by the inference container.

        The shape is part of the serving API — existing clients parse the
        `status`/`predictions`/`debug_image` keys.
        """
        if isinstance(prediction, list):
            predictions_json = [single.to_dict() for single in prediction]
        else:
            predictions_json = prediction.to_dict()
        response = {"status": "success", "predictions": predictions_json}
        if include_debug:
            debug_image = self.get_annotated_image(image, prediction)
            if debug_image is None:
                response["debug_image"] = None
            else:
                response["debug_image"] = image_to_base64(debug_image)
        return response

    def dump(self, dir: str):
        metadata = {
            "model_type": self.model_type,
            "classes": self.classes,
            "model_file": os.path.basename(self.model_path),
        }
        metadata_path = get_metadata_path(dir)
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)
        # model is already saved in the model_path
        return [metadata_path, self.model_path]
