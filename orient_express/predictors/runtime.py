"""ONNX Runtime session construction: devices and execution providers."""

import subprocess

import onnxruntime as ort


class Device:
    """Valid values for the `device` parameter across the library."""

    CPU = "cpu"
    CUDA = "cuda"


_DEVICE_TO_PROVIDER = {
    Device.CPU: "CPUExecutionProvider",
    Device.CUDA: "CUDAExecutionProvider",
}


def _preload_gpu_dlls():
    """Load CUDA/cuDNN from their pip wheels so ORT's dlopen finds them."""
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()


def _build_providers(device: str, provider_options: dict | None):
    if device == Device.CPU:
        return ["CPUExecutionProvider"]
    if device == Device.CUDA:
        _preload_gpu_dlls()
        # HEURISTIC picks conv algos without benchmarking every candidate.
        # Measured on our RF-DETR models: steady state within noise of
        # EXHAUSTIVE; it only guards warmup time on conv-heavy models.
        options = {"cudnn_conv_algo_search": "HEURISTIC", **(provider_options or {})}
        return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
    raise ValueError(
        f"Unknown device '{device}'. Supported: {', '.join(_DEVICE_TO_PROVIDER)}."
    )


def create_session(model_path: str, device: str, provider_options: dict | None = None):
    """Create an ORT InferenceSession for `device`, failing loudly on fallback."""
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_reuse = True

    providers = _build_providers(device, provider_options)
    session = ort.InferenceSession(
        model_path, providers=providers, sess_options=session_options
    )

    # ORT silently falls back to the CPU EP when a GPU provider can't load
    # (missing CUDA libs, wrong wheel) — in production that is a 10-50x
    # slowdown that looks like a working deployment. Fail loudly.
    if device != Device.CPU:
        active = session.get_providers()[0]
        wanted = _DEVICE_TO_PROVIDER[device]
        if active != wanted:
            raise RuntimeError(
                f"Requested device '{device}' ({wanted}) but onnxruntime "
                f"activated {active}. Check that the GPU onnxruntime build is "
                f"installed and a GPU is visible.{_driver_hint()}"
            )

    return session


def _driver_hint() -> str:
    """Report the local NVIDIA driver, the usual cause of a GPU-EP load failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        driver = out.stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001 - diagnosis only
        return " nvidia-smi found no usable NVIDIA driver on this machine."
    return f" Detected NVIDIA driver {driver}."
