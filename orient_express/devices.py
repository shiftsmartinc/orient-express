"""Device names, importable without an inference runtime.

These live outside `predictors/` because `predictors.runtime` imports
onnxruntime at module scope, while `vertex.py` — which validates the device
a deployment asks for — must stay importable for vertex-only installs that
carry no inference extra at all.
"""


class Device:
    """Valid values for the `device` parameter across the library."""

    CPU = "cpu"
    CUDA = "cuda"
    TENSORRT = "tensorrt"
    TENSORRT_FP16 = "tensorrt-fp16"
    TENSORRT_BF16 = "tensorrt-bf16"


# Devices that compile a TensorRT engine, and so need an optimization
# profile (see predictors.runtime.trt_profile_for_batch).
TENSORRT_DEVICES = (Device.TENSORRT, Device.TENSORRT_FP16, Device.TENSORRT_BF16)

ALL_DEVICES = (Device.CPU, Device.CUDA, *TENSORRT_DEVICES)
