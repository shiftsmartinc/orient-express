"""ONNX Runtime session construction: devices, providers, TRT engine caches."""

import ctypes
import glob
import hashlib
import logging
import os
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from threading import Event, Lock, Thread

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is not installed. Install orient_express with an "
        "inference extra: pip install 'orient_express[cpu]' (CPU), "
        "'orient_express[cuda]' (NVIDIA GPU), or 'orient_express[tensorrt]' "
        "(NVIDIA GPU + TensorRT)."
    ) from e

from ..utils.paths import get_cache_dir

_DEVICE_TO_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "tensorrt-fp16": "TensorrtExecutionProvider",
}


def parse_trt_profile_shapes(spec: str) -> dict[str, list[int]]:
    """Parse ORT's TRT profile syntax: 'images:1x576x576x3,target_sizes:1x2'."""
    shapes = {}
    for part in spec.split(","):
        name, _, dims = part.strip().rpartition(":")
        try:
            shapes[name] = [int(d) for d in dims.split("x")]
            if not name:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"Malformed TRT profile entry {part.strip()!r} in {spec!r}; "
                "expected 'input_name:1x576x576x3,other_input:1x2'."
            ) from None
    return shapes


def _preload_gpu_dlls():
    """Load CUDA/cuDNN from their pip wheels so ORT's dlopen finds them."""
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()


def _preload_tensorrt_libs():
    """Make libnvinfer visible to ORT's TensorRT provider.

    The pip tensorrt wheels keep their shared libraries inside the
    tensorrt_libs package, off the default loader path; loading them
    RTLD_GLOBAL up front means users don't need LD_LIBRARY_PATH.
    """
    try:
        import tensorrt_libs
    except ImportError as e:
        raise ImportError(
            "device='tensorrt' requires the TensorRT extra: "
            "pip install 'orient_express[tensorrt]'"
        ) from e
    lib_dir = os.path.dirname(tensorrt_libs.__file__)
    for pattern in (
        "libnvinfer.so.*",
        "libnvinfer_plugin.so.*",
        "libnvonnxparser.so.*",
    ):
        for lib in sorted(glob.glob(os.path.join(lib_dir, pattern))):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)


def _trt_version() -> str:
    for dist in ("tensorrt", "tensorrt-cu13", "tensorrt-cu12"):
        try:
            return _package_version(dist)
        except PackageNotFoundError:
            continue
    return "unknown"


def trt_cache_scope(model_path: str, provider_options: dict | None, fp16: bool) -> str:
    """Relative cache path unique to (model bytes, runtimes, profile, precision).

    A serialized TRT engine is only valid for the exact model, TensorRT and
    ORT versions, and optimization profile it was built with (ORT detects
    profile mismatches and rebuilds; version mismatches it does NOT detect —
    its docs require cleaning the cache manually). Scoping the cache
    directory by all four means an entry is written once and never churned,
    workers download only entries for exactly what they are loading, and
    stale entries are simply never fetched again.
    """
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    scope = f"{h.hexdigest()[:16]}-ort{ort.__version__}-trt{_trt_version()}"
    profile = "|".join(
        str((provider_options or {}).get(key, ""))
        for key in (
            "trt_profile_min_shapes",
            "trt_profile_opt_shapes",
            "trt_profile_max_shapes",
        )
    )
    if profile != "||":
        scope += "-p" + hashlib.sha256(profile.encode()).hexdigest()[:8]
    return f"{scope}/{'fp16' if fp16 else 'fp32'}"


def trt_engine_cache_dir(scope: str) -> str:
    """Local directory for TensorRT engine + timing caches.

    `scope` (from trt_cache_scope) isolates model, runtime versions, profile
    config and precision from each other. Override the root with
    ORIENT_EXPRESS_TRT_CACHE_DIR.
    """
    root = os.environ.get("ORIENT_EXPRESS_TRT_CACHE_DIR") or os.path.join(
        get_cache_dir(), "trt-engine-cache"
    )
    path = os.path.join(root, *scope.split("/"))
    os.makedirs(path, exist_ok=True)
    return path


def _build_providers(
    device: str, provider_options: dict | None, trt_scope: str | None = None
):
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        _preload_gpu_dlls()
        # HEURISTIC picks conv algos without benchmarking every candidate.
        # Measured on our RF-DETR models: steady state within noise of
        # EXHAUSTIVE; it only guards warmup time on conv-heavy models.
        options = {"cudnn_conv_algo_search": "HEURISTIC", **(provider_options or {})}
        return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
    if device in ("tensorrt", "tensorrt-fp16"):
        _preload_gpu_dlls()
        _preload_tensorrt_libs()
        fp16 = device.endswith("fp16")
        cache = trt_engine_cache_dir(trt_scope)
        options = {
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache,
            **(provider_options or {}),
        }
        return [
            ("TensorrtExecutionProvider", options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    raise ValueError(
        f"Unknown device '{device}'. Supported: {', '.join(_DEVICE_TO_PROVIDER)}."
    )


class _TrtCacheGcsSync:
    """Mirror the local TRT engine cache to a GCS prefix.

    Set ORIENT_EXPRESS_TRT_CACHE_GCS=gs://bucket/prefix and short-lived
    workers (e.g. Vertex pipelines on a fixed GPU type) download prebuilt
    engines instead of spending minutes compiling. Engines build lazily on
    the first predict() with a new shape, so after such predicts an upload
    sweep runs on a background thread, off the inference hot path.

    The object prefix carries the trt_cache_scope suffix (model hash, ORT and
    TRT versions, profile config, precision), so one bucket prefix can serve
    every model and pool: workers download only the entries for exactly what
    they load, and entries orphaned by model or version bumps are never
    fetched again (expire them with a GCS lifecycle rule). GPU architectures
    coexist inside one scope — ORT keys the engine filenames by SM arch.
    Sync failures degrade to a local build with a warning — the cache is an
    optimization, never a correctness dependency.
    """

    def __init__(self, gcs_prefix: str, local_dir: str):
        from ..utils import gs

        self._gs = gs
        self.prefix = gcs_prefix.rstrip("/")
        self.local_dir = local_dir
        self._synced: dict[str, float] = {}
        # per-request GCS timeout; timeouts surface as the standard warning
        self._timeout = float(os.environ.get("ORIENT_EXPRESS_TRT_CACHE_TIMEOUT", "60"))
        self._wake = Event()
        self._worker: Thread | None = None
        self._start_lock = Lock()

    def download(self):
        try:
            from google.cloud import storage

            bucket_name, path = self._gs.parse_gcs_url(self.prefix)
            bucket = storage.Client().bucket(bucket_name)
            for blob in bucket.list_blobs(prefix=path + "/"):
                name = os.path.basename(blob.name)
                if not name:
                    continue
                local = os.path.join(self.local_dir, name)
                if not os.path.exists(local):
                    blob.download_to_filename(local, timeout=self._timeout)
                self._synced[name] = os.path.getmtime(local)
        except Exception as e:  # noqa: BLE001 - cache is best-effort
            logging.warning(f"TRT cache download from {self.prefix} failed: {e}")

    def schedule_upload(self):
        """Wake the background uploader; repeat calls coalesce into one sweep.

        Called after a predict with a first-seen input shape (the only runs
        that can build engines), so uploads never block inference. The
        worker is a daemon thread: a slow or hung upload never delays
        process exit — an upload killed mid-flight is harmless (GCS object
        creation is atomic) and the next worker's sweep re-pushes it.
        """
        with self._start_lock:
            if self._worker is None:
                self._worker = Thread(
                    target=self._upload_loop, daemon=True, name="trt-cache-sync"
                )
                self._worker.start()
        self._wake.set()

    def _upload_loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            self.upload_new()

    def upload_new(self):
        # A stat sweep over the handful of cache files finds anything new or
        # rewritten. The recorded mtime is taken before the upload, so a file
        # modified mid-upload (e.g. the timing cache during a concurrent
        # engine build) looks dirty again on the next sweep and is re-pushed
        # clean — a torn object in GCS only ever costs a downstream rebuild.
        try:
            for name in os.listdir(self.local_dir):
                local = os.path.join(self.local_dir, name)
                if not os.path.isfile(local):
                    continue
                file_mtime = os.path.getmtime(local)
                if self._synced.get(name) == file_mtime:
                    continue
                self._gs.upload_file(
                    local, f"{self.prefix}/{name}", timeout=self._timeout
                )
                self._synced[name] = file_mtime
        except Exception as e:  # noqa: BLE001 - cache is best-effort
            logging.warning(f"TRT cache upload to {self.prefix} failed: {e}")


def create_session(model_path: str, device: str, provider_options: dict | None = None):
    """Create an ORT InferenceSession for `device`, failing loudly on fallback.

    Returns (session, trt_cache_sync); trt_cache_sync is a _TrtCacheGcsSync
    when device is a tensorrt variant and ORIENT_EXPRESS_TRT_CACHE_GCS is
    set, else None.
    """
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_reuse = True

    trt_scope = None
    if device.startswith("tensorrt"):
        trt_scope = trt_cache_scope(
            model_path, provider_options, fp16=device.endswith("fp16")
        )
    providers = _build_providers(device, provider_options, trt_scope)

    trt_cache_sync = None
    if trt_scope is not None:
        gcs_prefix = os.environ.get("ORIENT_EXPRESS_TRT_CACHE_GCS")
        if gcs_prefix:
            cache_dir = providers[0][1]["trt_engine_cache_path"]
            trt_cache_sync = _TrtCacheGcsSync(
                f"{gcs_prefix.rstrip('/')}/{trt_scope}", cache_dir
            )
            trt_cache_sync.download()

    session = ort.InferenceSession(
        model_path, providers=providers, sess_options=session_options
    )

    # ORT silently falls back to the CPU EP when a GPU provider can't load
    # (missing CUDA libs, wrong wheel) — in production that is a 10-50x
    # slowdown that looks like a working deployment. Fail loudly.
    if device != "cpu":
        active = session.get_providers()[0]
        wanted = _DEVICE_TO_PROVIDER[device]
        if active != wanted:
            raise RuntimeError(
                f"Requested device '{device}' ({wanted}) but onnxruntime "
                f"activated {active}. Check that the right extra is "
                f"installed ('orient_express[cuda]' / '[tensorrt]') and a "
                f"GPU is visible.{_driver_hint()}"
            )
    return session, trt_cache_sync


def _driver_hint() -> str:
    """Diagnose the most common GPU-EP load failure: driver too old.

    The cuda/tensorrt extras ship CUDA-13 user-space libraries, which need
    NVIDIA driver r580+; the system CUDA toolkit version is irrelevant.
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
            " NVIDIA's cuda-compat-13 package in the image, or switch to a"
            " CUDA-12 wheel stack."
        )
    return hint
