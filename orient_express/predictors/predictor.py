import os
import warnings
from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from PIL import Image

from ..utils.colors import generate_color_scheme
from ..utils.image_processor import image_to_array, image_to_base64
from ..utils.paths import get_metadata_path

IMAGE_ONNX_IMAGE_REPO = (
    "us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx"
)


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
    def from_dir(cls, dir: str, metadata: dict, device: str = "cpu") -> "Predictor":
        """Construct this predictor from a downloaded artifact directory."""
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
    model_type: str
    backend_model: type
    prediction_type: type

    def __init__(self, model_path: str, classes: dict[int, str], device: str = "cpu"):
        self.model = self.backend_model(model_path, device)
        self.color_scheme = generate_color_scheme(list(classes.values()))
        self.classes = classes
        self.model_path = model_path

    @classmethod
    def from_dir(cls, dir: str, metadata: dict, device: str = "cpu"):
        if "model_file" not in metadata:
            raise Exception("No model_file defined in metadata.yaml")
        if "classes" not in metadata:
            raise Exception("No classes defined in metadata.yaml")
        onnx_path = os.path.join(dir, metadata["model_file"])
        return cls(onnx_path, metadata["classes"], device)

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


class OnnxSessionWrapper:
    def __init__(self, onnx_path: str, device: str = "cpu"):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_reuse = True

        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            warnings.warn(
                f"Unknown device '{device}'. Defaulting to CPU. Supported devices: 'cpu', 'cuda'.",
                stacklevel=2,
            )
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=session_options
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        input_shape = self.session.get_inputs()[0].shape
        self.resolution = input_shape[1]
        self.img_size = (self.resolution, self.resolution)

    def collate_sizes(self, pil_images: list[Image.Image]):
        sizes = [[img.size[1], img.size[0]] for img in pil_images]
        return np.array(sizes, dtype=np.float32)

    def collate_images(self, pil_images: list[Image.Image]):
        images = [cv2.resize(image_to_array(img), self.img_size) for img in pil_images]
        return np.array(images)
