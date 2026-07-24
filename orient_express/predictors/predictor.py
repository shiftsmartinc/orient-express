import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

import cv2
import numpy as np
import yaml
from PIL import Image

from ..utils.colors import generate_color_scheme
from ..utils.image_processor import image_to_array, image_to_base64
from ..utils.paths import get_metadata_path
from .runtime import Device, create_session, parse_trt_profile_shapes

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

    ORT is the only backend: its TensorRT execution provider performs within
    ~2% of native TensorRT.

    Inference is split into three public stages so callers can pipeline:

        feed = predictor.preprocess(images)     # CPU: collate/resize
        outputs = predictor.infer(feed)         # GPU: session.run
        preds = predictor.postprocess(outputs, feed, **kwargs)  # CPU

    predict() composes the three. cv2.resize and session.run both release
    the GIL, so the CPU stages of one batch can overlap with inference on
    another.
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
        if device in (Device.TENSORRT, Device.TENSORRT_FP16) and provider_options:
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
        their own keyword arguments (e.g. confidence).
        """
        raise NotImplementedError

    def collate_sizes(self, pil_images: list[Image.Image]):
        sizes = [[img.size[1], img.size[0]] for img in pil_images]
        return np.array(sizes, dtype=np.float32)

    def collate_images(self, pil_images: list[Image.Image]):
        n = len(pil_images)
        batch = np.empty((n, self.resolution, self.resolution, 3), dtype=np.uint8)

        def collate_one(i):
            batch[i] = cv2.resize(image_to_array(pil_images[i]), self.img_size)

        # cv2.resize releases the GIL, so batches of full-size photos collate
        # ~3x faster on a thread pool; batches of small crops (e.g. from
        # build_vector_index) stay serial — task dispatch would dominate their
        # ~microsecond resizes. Calibrated empirically, see PR summary.
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
