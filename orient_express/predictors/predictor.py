import os
from abc import ABC, abstractmethod
import warnings

import yaml
import onnxruntime as ort
from PIL import Image
import cv2
import numpy as np

from ..utils.paths import get_metadata_path
from ..utils.colors import generate_color_scheme
from ..utils.image_processor import image_to_array


class Predictor(ABC):
    model_type: str
    model_path: str

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

    def get_serving_container_image_uri(self):
        return "us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx:v2.1.2"

    def get_serving_container_health_route(self, model_name):
        return f"/v1/models/{model_name}"

    def get_serving_container_predict_route(self, model_name):
        return f"/v1/models/{model_name}:predict"

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
                f"Unknown device '{device}'. Defaulting to CPU. Supported devices: 'cpu', 'cuda'."
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
