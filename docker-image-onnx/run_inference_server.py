import logging
import logging.config
import os
from concurrent.futures import ThreadPoolExecutor

from kserve import Model, ModelServer
from PIL import Image as PILImage

from orient_express.predictors import (
    TENSORRT_DEVICES,
    Device,
    ImagePredictor,
    get_predictor,
    gpu_available,
)
from orient_express.serving import (
    build_predict_kwargs,
    decode_input,
    download_image,
    runtime_info_response,
)
from orient_express.utils.image_processor import fix_rotation
from orient_express.vertex import (
    ARTIFACT_DIR,
    download_artifacts,
    get_deployed_model_device,
)

# Largest batch a TensorRT engine is built for. Requests above it are split
# (see predict); raising it costs engine build time and GPU memory.
TRT_MAX_BATCH_SIZE = 8


class OnnxImageModel(Model):
    def __init__(self, name: str, artifacts_path: str):
        super().__init__(name)
        self.name = name
        self.artifacts_path = artifacts_path
        self.model = None
        self.device = None

    def load(self):
        logging.info(f"[{self.name}] loading model from GCS")
        download_dir = os.path.join(ARTIFACT_DIR, self.name)
        download_artifacts(download_dir, self.artifacts_path)
        # One serving image for every runtime; the device is chosen at
        # DEPLOY time, never upload time. deploy_to_endpoint(device=...)
        # stamps a device= token on the DeployedModel, read back here. With
        # no token the hardware decides — cuda on a GPU-attached deployment,
        # else cpu — so an accelerator is never left idle.
        #
        # ORIENT_EXPRESS_DEVICE exists solely to exercise a device when this
        # image runs OUTSIDE Vertex (make run_local_image_onnx), where there
        # is no DeployedModel to carry a token. Deliberately undocumented:
        # in production the deployment is the only place to say this.
        device = (
            os.environ.get("ORIENT_EXPRESS_DEVICE")
            or get_deployed_model_device()
            or (Device.CUDA if gpu_available() else Device.CPU)
        )
        logging.info(f"[{self.name}] serving device: {device}")
        self.device = device
        # TensorRT needs an optimization profile and a deployment supplies
        # only a device string; trt_batch is the whole of what serving can
        # say about shapes. opt=1 because the dominant online request is a
        # single image; predict() splits requests over TRT_MAX_BATCH_SIZE.
        kwargs = (
            {"trt_batch": (1, 1, TRT_MAX_BATCH_SIZE)}
            if device in TENSORRT_DEVICES
            else {}
        )
        # Graph optimizations OFF here, unlike the library default. This
        # image is pinned to the CUDA-12 onnxruntime line (Vertex's L4
        # driver cannot run CUDA 13), and onnxruntime 1.24.x fuses
        # DINOv3-backbone graphs into something that reads uninitialized
        # memory — the same image scores differently in every replica.
        # Disabling fusion restores the exported graph's numbers, which
        # match ORT 1.27 to ~1e-6. Revisit if this image ever moves to a
        # CUDA-13 onnxruntime: 1.27+ has no such bug.
        self.model = get_predictor(
            download_dir, device, graph_optimizations=False, **kwargs
        )
        self.warmup()
        self.ready = True
        logging.info(f"{self.name} loaded successfully")
        return self

    def warmup(self):
        # The first ONNX inference pays one-time allocation/setup costs; run
        # it here so the first real request doesn't.
        if not isinstance(self.model, ImagePredictor):
            return
        try:
            dummy = PILImage.new("RGB", (64, 64))
            kwargs = build_predict_kwargs(self.model.predict, {})
            self.model.predict([dummy], **kwargs)
            logging.info(f"[{self.name}] warmup inference complete")
        except Exception:
            logging.exception(f"[{self.name}] warmup inference failed (non-fatal)")

    def predict(self, inputs, *args, **kwargs):
        logging.info(f"[{self.name}] executing prediction")

        assert self.model is not None

        try:
            decoded_input = decode_input(inputs)
            instances = decoded_input["instances"]
            parameters = decoded_input.get("parameters", {}) or {}
        except Exception as e:
            logging.exception(f"[{self.name}] failed to decode input: {e}\n{inputs}")
            return {"error": "Failed to decode input"}

        if parameters.get("runtime_info"):
            # deploy-time verification (see VertexModel.deploy_to_endpoint):
            # report the provider the live ORT session actually activated
            return runtime_info_response(self.model, self.device)

        include_debug = bool(parameters.get("debug_image", True))
        predict_kwargs = build_predict_kwargs(self.model.predict, parameters)

        predictions: list[dict] = [{} for _ in instances]

        images = []
        image_idxs = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(download_image, instance["image"])
                for instance in instances
            ]

            for img_idx, future in enumerate(futures):
                try:
                    image = future.result()
                    image = fix_rotation(image)
                    images.append(image)
                    image_idxs.append(img_idx)
                except Exception as e:
                    logging.exception(
                        f"[{self.name}] failed to download image {img_idx}: {e}"
                    )
                    predictions[img_idx] = {"status": "failed to download image"}

        # A TensorRT device only accepts batches inside its optimization
        # profile; requests larger than the profile's max batch are split
        # (predictions are per-image, so chunking changes nothing else).
        max_batch = getattr(self.model, "max_batch_size", None)
        if max_batch and len(images) > max_batch:
            model_predictions = []
            for start in range(0, len(images), max_batch):
                model_predictions.extend(
                    self.model.predict(
                        images[start : start + max_batch], **predict_kwargs
                    )
                )
        else:
            model_predictions = self.model.predict(images, **predict_kwargs)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.model.to_response, image, prediction, include_debug
                )
                for image, prediction in zip(images, model_predictions, strict=True)
            ]

            for pred_idx, future in enumerate(futures):
                img_idx = image_idxs[pred_idx]
                try:
                    predictions[img_idx] = future.result()
                except Exception as e:
                    logging.exception(
                        f"[{self.name}] failed to build response {img_idx}: {e}"
                    )
                    predictions[img_idx] = {"status": "failed to get debug image"}

        return {"predictions": predictions}


if __name__ == "__main__":
    logging.config.fileConfig("logging.conf")

    try:
        storage_uri = os.environ["AIP_STORAGE_URI"]
        model_name = os.environ["MODEL_NAME"]

        model = OnnxImageModel(model_name, storage_uri)
        model.load()

        model_server = ModelServer(http_port=8080, workers=1)
        model_server.start([model])
    except Exception as e:
        logging.exception("Failed to start model")
        raise e
