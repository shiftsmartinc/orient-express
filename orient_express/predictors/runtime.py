"""ONNX Runtime session construction: devices and execution providers."""

import ctypes
import glob
import os
import subprocess


def _preload_cuda_runtime():
    """Load libcudart from the nvidia wheels before onnxruntime is imported.

    The GPU onnxruntime build links libcudart directly, so `import
    onnxruntime` fails outright when the dynamic loader can't find it — and
    ORT's own preload_dlls() is defined past that failing import, so it
    cannot help. The cuda extras ship libcudart inside the nvidia wheels,
    which sit off the default loader path; loading it RTLD_GLOBAL here is
    what lets an image with no system CUDA install (e.g. a slim python base
    image) work without LD_LIBRARY_PATH. Loading by SONAME satisfies the
    extension's link at import. A CPU-only install has no such wheel and
    falls through untouched.
    """
    import sys

    for entry in sys.path:
        if not entry:
            continue
        pattern = os.path.join(entry, "nvidia", "*", "lib", "libcudart.so.*")
        for lib in sorted(glob.glob(pattern), reverse=True):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


_preload_cuda_runtime()

# imported after the preload above, which the GPU build depends on
try:
    import onnxruntime as ort  # noqa: E402
except ImportError as e:
    raise ImportError(
        "onnxruntime is not installed. Install orient_express with an "
        "inference extra: pip install 'orient_express[cpu]' (CPU) or "
        "'orient_express[cuda]' (NVIDIA GPU)."
    ) from e


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
                f"activated {active}. Check that the right extra is "
                f"installed ('orient_express[cuda]') and a GPU is "
                f"visible.{_driver_hint()}"
            )

    return session


def _driver_hint() -> str:
    """Diagnose the most common GPU-EP load failure: driver too old.

    The cuda extra ships CUDA-13 user-space libraries, which need NVIDIA
    driver r580+; the system CUDA toolkit version is irrelevant.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        driver = out.stdout.strip().splitlines()[0].strip()
        major = int(driver.split(".")[0])
    except Exception:  # noqa: BLE001 - diagnosis only
        return " nvidia-smi found no usable NVIDIA driver on this machine."
    hint = f" Detected NVIDIA driver {driver}."
    if major < 580:
        hint += (
            " These wheels are CUDA-13 builds, which need driver r580+."
            " Fix: upgrade the driver, or (datacenter GPUs, e.g. L4) install"
            " NVIDIA's cuda-compat-13 package in the image, or switch to the"
            " cuda12 extra."
        )
    return hint
