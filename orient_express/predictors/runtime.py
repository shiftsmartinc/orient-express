"""ONNX Runtime session construction: devices, providers, TRT engine caches."""

import ctypes
import glob
import hashlib
import logging
import os
import re
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from threading import Event, Lock, Thread

from ..devices import DEVICE_TO_PROVIDER, TENSORRT_DEVICES, Device
from ..utils.paths import get_cache_dir
from ..utils.retry import get_gcs_retry_policy


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
        "inference extra: pip install 'orient_express[cpu]' (CPU), "
        "'orient_express[cuda]' (NVIDIA GPU), or 'orient_express[tensorrt]' "
        "(NVIDIA GPU + TensorRT)."
    ) from e


# Engine precision per TRT device — also the cache-scope leaf directory.
_TRT_PRECISION = {
    Device.TENSORRT: "fp32",
    Device.TENSORRT_FP16: "fp16",
    Device.TENSORRT_BF16: "bf16",
}

# Provider options reserved for device selection (see _build_providers).
_PRECISION_OPTIONS = frozenset({"trt_fp16_enable", "trt_bf16_enable"})


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


def gpu_available() -> bool:
    """Cheap probe: is an NVIDIA GPU visible to this process?

    Checks device nodes first (covers containers — Vertex injects
    /dev/nvidia* into GPU-attached deployments) and falls back to
    nvidia-smi (covers hosts).
    """
    if glob.glob("/dev/nvidia[0-9]*"):
        return True
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:  # noqa: BLE001 - probe only
        return False


def _normalize_trt_batch(trt_batch: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """(min, opt, max) from `trt_batch`; a bare N means (1, N, N).

    min stays 1 so a ragged final batch never falls out of profile, while
    opt tunes for N — the shape a batch pipeline actually sends.
    """
    if isinstance(trt_batch, int) and not isinstance(trt_batch, bool):
        if trt_batch < 1:
            raise ValueError(f"trt_batch must be >= 1, got {trt_batch}")
        return (1, trt_batch, trt_batch)
    try:
        low, opt, high = trt_batch
        low, opt, high = int(low), int(opt), int(high)
    except (TypeError, ValueError):
        raise ValueError(
            f"trt_batch must be an int or a (min, opt, max) triple, got {trt_batch!r}"
        ) from None
    if not 1 <= low <= opt <= high:
        raise ValueError(
            f"trt_batch must satisfy 1 <= min <= opt <= max, got {trt_batch!r}"
        )
    return (low, opt, high)


def trt_profile_for_batch(model_path: str, trt_batch) -> dict:
    """trt_profile_* specs that range only the model's batch dimension.

    TensorRT demands complete shapes for every input, but the batch range
    is the only part a caller can meaningfully choose: every other
    dimension is pinned by the graph, and TRT discards whatever is written
    there (measured — a profile claiming 999x999 still runs a 576x576
    input). So take the batch bounds from the caller and transcribe the
    rest off the model. The throwaway CPU session skips graph
    optimization; only the declared input shapes are read from it.

    The batch dimension is found by its ONNX symbolic name rather than by
    position, which is what makes multi-input models work: detection's
    `images` and `target_sizes` share one symbolic dim, so a single bound
    fans out to every input carrying it, at whatever axis it sits. Two or
    more distinct symbolic names are ambiguous — which one is the batch? —
    and a fully static graph cannot honour a range at all; both raise
    rather than quietly profiling something the caller did not ask for.
    """
    low, opt, high = _normalize_trt_batch(trt_batch)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"], sess_options=session_options
    )
    inputs = [(inp.name, inp.shape) for inp in session.get_inputs()]
    model = os.path.basename(model_path)
    symbolic = {d for _, shape in inputs for d in shape if not isinstance(d, int)}

    if len(symbolic) > 1:
        raise ValueError(
            f"{model} has more than one dynamic dimension "
            f"({', '.join(sorted(symbolic))}); which one is the batch cannot "
            "be inferred. Pass explicit trt_profile_min/opt/max_shapes "
            "instead."
        )
    if not symbolic:
        pinned = inputs[0][1][0] if inputs and inputs[0][1] else None
        if (low, opt, high) != (pinned, pinned, pinned):
            raise ValueError(
                f"{model} is a static export pinned at batch {pinned}, so "
                f"trt_batch={trt_batch!r} cannot be honoured. Re-export with a "
                "dynamic batch dimension (see MODEL_REQUIREMENTS.md), or pass "
                f"trt_batch={pinned} to profile the batch it does support."
            )

    def spec(batch: int) -> str:
        return ",".join(
            name
            + ":"
            + "x".join(str(d) if isinstance(d, int) else str(batch) for d in shape)
            for name, shape in inputs
        )

    return {
        "trt_profile_min_shapes": spec(low),
        "trt_profile_opt_shapes": spec(opt),
        "trt_profile_max_shapes": spec(high),
    }


def apply_trt_batch(
    model_path: str, device: str, trt_batch, provider_options: dict | None
) -> dict:
    """Fold `trt_batch` into provider_options as profile specs."""
    if device not in TENSORRT_DEVICES:
        raise ValueError(
            f"trt_batch selects a TensorRT optimization profile and does not "
            f"apply to device='{device}'."
        )
    given = set(_PROFILE_OPTIONS) & set(provider_options or {})
    if given:
        raise ValueError(
            f"pass either trt_batch or explicit profile shapes, not both (got "
            f"trt_batch and {', '.join(sorted(given))})."
        )
    return {**trt_profile_for_batch(model_path, trt_batch), **(provider_options or {})}


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


def _crc32c_b64(path: str) -> str:
    """crc32c of a file, base64-encoded — the format GCS blob metadata uses."""
    import base64

    import google_crc32c

    checksum = google_crc32c.Checksum()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode()


def _local_sm_tags() -> set[str] | None:
    """SM tokens for the local GPUs (e.g. {'sm120'}), or None if unknown.

    Matches the arch token ORT embeds in engine/timing cache filenames, so
    cache downloads can skip other GPU generations' engines. None (no
    nvidia-smi, unexpected output) means no filtering — downloading too
    much is safe, skipping wrongly is not.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        tags = {
            "sm" + cap.strip().replace(".", "")
            for cap in out.stdout.strip().splitlines()
            if cap.strip()
        }
        return tags or None
    except Exception:  # noqa: BLE001 - optimization only
        return None


def _trt_version() -> str:
    for dist in ("tensorrt", "tensorrt-cu13", "tensorrt-cu12"):
        try:
            return _package_version(dist)
        except PackageNotFoundError:
            continue
    return "unknown"


_PROFILE_OPTIONS = (
    "trt_profile_min_shapes",
    "trt_profile_opt_shapes",
    "trt_profile_max_shapes",
)

# Cache plumbing that never influences the compiled engine. Every OTHER
# provider option the caller passes is assumed to affect the build (e.g.
# trt_layer_norm_fp32_fallback, trt_builder_optimization_level) and splits
# the cache scope: a redundant rebuild is cheap, silently reusing an engine
# built under different options is not.
_SCOPE_IRRELEVANT_OPTIONS = frozenset(
    {
        "trt_engine_cache_enable",
        "trt_engine_cache_path",
        "trt_timing_cache_enable",
        "trt_timing_cache_path",
    }
)


def trt_cache_scope(
    model_path: str, provider_options: dict | None, precision: str
) -> str:
    """Relative cache path unique to (model bytes, runtimes, options, precision).

    A serialized TRT engine is only valid for the exact model, TensorRT and
    ORT versions, optimization profile, and engine-affecting provider
    options it was built with (ORT detects profile mismatches and rebuilds;
    version mismatches it does NOT detect — its docs require cleaning the
    cache manually). Scoping the cache directory by all of them means an
    entry is written once and never churned, workers download only entries
    for exactly what they are loading, and stale entries are simply never
    fetched again. NVIDIA_TF32_OVERRIDE joins the key when set: TRT bakes
    TF32 tactics into the engine, and a cached engine built under one
    setting hard-fails to load under another instead of rebuilding.
    """
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    scope = f"{h.hexdigest()[:16]}-ort{ort.__version__}-trt{_trt_version()}"
    options = provider_options or {}
    profile = "|".join(str(options.get(key, "")) for key in _PROFILE_OPTIONS)
    if profile != "||":
        scope += "-p" + hashlib.sha256(profile.encode()).hexdigest()[:8]
    extra = [
        (key, options[key])
        for key in sorted(options)
        if key not in _PROFILE_OPTIONS and key not in _SCOPE_IRRELEVANT_OPTIONS
    ]
    tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    if tf32_override is not None:
        extra.append(("env:NVIDIA_TF32_OVERRIDE", tf32_override))
    if extra:
        joined = "|".join(f"{key}={value}" for key, value in extra)
        scope += "-o" + hashlib.sha256(joined.encode()).hexdigest()[:8]
    return f"{scope}/{precision}"


def trt_engine_cache_dir(scope: str) -> str:
    """Local directory for TensorRT engine + timing caches.

    `scope` (from trt_cache_scope) isolates model, runtime versions, profile
    config and precision from each other. Override the root with
    ORIENT_EXPRESS_TRT_CACHE_DIR. Least-recently-used scopes are evicted
    when the cache exceeds ORIENT_EXPRESS_TRT_CACHE_MAX_BYTES (default
    20GB, 0 disables).
    """
    root = os.environ.get("ORIENT_EXPRESS_TRT_CACHE_DIR") or os.path.join(
        get_cache_dir(), "trt-engine-cache"
    )
    path = os.path.join(root, *scope.split("/"))
    os.makedirs(path, exist_ok=True)
    _prune_cache_root(root, keep=path)
    return path


# The exact shape of a scope dir minted by trt_cache_scope():
# <16-hex model hash>-ort<ver>-trt<ver>[-p<8-hex profile hash>]. Pruning
# recognizes candidates by this shape, so it can only ever delete
# directories this library created — never the root itself, loose files,
# or foreign dirs a user co-located under a shared cache root.
_SCOPE_DIR_RE = re.compile(r"^[0-9a-f]{16}-ort.+-trt.+$")
_PRECISION_DIRS = ("fp16", "fp32", "bf16")


def _prune_cache_root(root: str, keep: str):
    """Evict least-recently-used cache scopes once the root exceeds its cap.

    Every model/profile/option variation mints a scope dir holding a
    ~100MB+ engine forever; without a cap the cache root grows unbounded.
    Only fp16/fp32 leaves of scope-shaped dirs (per _SCOPE_DIR_RE) are
    measured and evicted. Eviction is safe by construction: a live session
    already deserialized its engine into memory (the file is never
    re-read), and a delete racing another process's load window just
    degrades to one rebuild. The scope currently being resolved is never
    evicted. ORT touches file mtimes on every load, which here is a
    feature: mtime IS last-used.
    """
    default = 20 * 1024**3
    raw = os.environ.get("ORIENT_EXPRESS_TRT_CACHE_MAX_BYTES", str(default))
    try:
        cap = int(raw)
    except ValueError:
        logging.warning(
            f"ORIENT_EXPRESS_TRT_CACHE_MAX_BYTES={raw!r} is not an int; "
            f"using default {default}"
        )
        cap = default
    if cap <= 0:
        return
    try:
        keep = os.path.normpath(keep)
        total = 0
        units = []  # (last_used, size, leaf_path) per fp16/fp32 leaf dir
        for scope in os.scandir(root):
            if not scope.is_dir() or not _SCOPE_DIR_RE.match(scope.name):
                continue
            for leaf in os.scandir(scope.path):
                if not leaf.is_dir() or leaf.name not in _PRECISION_DIRS:
                    continue
                stats = [f.stat() for f in os.scandir(leaf.path) if f.is_file()]
                if not stats:
                    continue
                size = sum(st.st_size for st in stats)
                total += size
                if os.path.normpath(leaf.path) != keep:
                    units.append((max(st.st_mtime for st in stats), size, leaf.path))
        if total <= cap:
            return
        units.sort()
        for _, size, leaf_path in units:
            shutil.rmtree(leaf_path, ignore_errors=True)
            try:
                os.rmdir(os.path.dirname(leaf_path))  # scope dir, if now empty
            except OSError:
                pass
            logging.info(f"TRT cache over {cap} bytes: evicted {leaf_path}")
            total -= size
            if total <= cap:
                return
    except Exception as e:  # noqa: BLE001 - GC is best-effort
        logging.warning(f"TRT cache pruning under {root} failed: {e}")


def _build_providers(
    device: str, provider_options: dict | None, trt_scope: str | None = None
):
    if device == Device.CPU:
        return ["CPUExecutionProvider"]
    if device == Device.CUDA:
        _preload_gpu_dlls()
        # HEURISTIC picks conv algos without benchmarking every candidate,
        # which mainly cuts warmup time: for detectors light on convolutions
        # (e.g. RF-DETR) steady state is indistinguishable from EXHAUSTIVE
        # search.
        options = {"cudnn_conv_algo_search": "HEURISTIC", **(provider_options or {})}
        return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
    if device in TENSORRT_DEVICES:
        missing = [
            key for key in _PROFILE_OPTIONS if not (provider_options or {}).get(key)
        ]
        if missing:
            raise ValueError(
                f"device='{device}' requires an explicit TensorRT optimization "
                "profile: set trt_profile_min_shapes, trt_profile_opt_shapes and "
                f"trt_profile_max_shapes in provider_options (missing: "
                f"{', '.join(missing)}), covering every input shape you will "
                "send, e.g. 'images:1x576x576x3,target_sizes:1x2'. Without one, "
                "TensorRT compiles an engine for the first shape it sees and any "
                "other shape forces a multi-minute rebuild. Note: ORT may "
                "additionally require entries for internal partition inputs — "
                "it names them in its session-init error."
            )
        # Engine precision is chosen by the device string, never by provider
        # options: a device named one precision quietly building another (or
        # a mixed fp16+bf16 engine) is exactly the ambiguity the dedicated
        # devices exist to remove. Reject rather than silently override so
        # the caller's mistaken intent surfaces.
        conflicting = _PRECISION_OPTIONS & set(provider_options or {})
        if conflicting:
            raise ValueError(
                f"provider_options may not set {', '.join(sorted(conflicting))}: "
                "engine precision is selected by the device string — use "
                f"device='{Device.TENSORRT}' (fp32), '{Device.TENSORRT_FP16}' "
                f"or '{Device.TENSORRT_BF16}' instead."
            )
        _preload_gpu_dlls()
        _preload_tensorrt_libs()
        cache = trt_engine_cache_dir(trt_scope)
        options = {
            "trt_fp16_enable": device == Device.TENSORRT_FP16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache,
            **(provider_options or {}),
        }
        if device == Device.TENSORRT_BF16:
            # bf16 keeps fp32's exponent range at 16-bit throughput — the
            # 16-bit mode for models whose activations overflow fp16's 65504
            # max (e.g. DINOv3 backbones, which carry ~1.5e5 residual-stream
            # activations and NaN under fp16). Set only for this device: ORT
            # builds predating the option reject unknown provider keys, and
            # fp32/fp16 users should not pay that compatibility cost.
            options["trt_bf16_enable"] = True
        return [
            ("TensorrtExecutionProvider", options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    raise ValueError(
        f"Unknown device '{device}'. Supported: {', '.join(DEVICE_TO_PROVIDER)}."
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

    # timeout applies per upload request; chunking keeps that a per-16MB
    # bound instead of a whole-engine bound (a 118MB engine on a shared
    # uplink exceeds 60s and would otherwise never reach GCS)
    _UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024

    def __init__(self, gcs_prefix: str, local_dir: str):
        from ..utils import gs

        self._gs = gs
        self.prefix = gcs_prefix.rstrip("/")
        self.local_dir = local_dir
        # name -> crc32c (base64, as GCS reports it). Content identity, not
        # mtime: ORT rewrites every cache file on every session load, so
        # mtimes mark everything dirty and each worker would re-upload the
        # whole engine every run.
        self._synced: dict[str, str] = {}
        # caps each GCS call INCLUDING retries (the client's default retry
        # policy would otherwise keep retrying timeouts for 120s, so an
        # outage would stall every cold start that long regardless of this
        # setting); failures surface as the standard warning
        self._timeout = float(os.environ.get("ORIENT_EXPRESS_TRT_CACHE_TIMEOUT", "60"))
        self._wake = Event()
        self._worker: Thread | None = None
        self._start_lock = Lock()

    def download(self):
        try:
            from google.cloud import storage

            bucket_name, path = self._gs.parse_gcs_url(self.prefix)
            bucket = storage.Client().bucket(bucket_name)
            retry_policy = get_gcs_retry_policy(timeout=self._timeout)
            sm_tags = _local_sm_tags()
            for blob in bucket.list_blobs(
                prefix=path + "/", timeout=self._timeout, retry=retry_policy
            ):
                name = os.path.basename(blob.name)
                if not name:
                    continue
                # ORT keys engine/timing filenames by SM arch; another GPU
                # generation's engine can never load here, so don't spend
                # cold-start time downloading it (mixed fleets share scopes)
                arch = re.search(r"_sm(\d+)", name)
                if sm_tags is not None and arch and arch.group(0)[1:] not in sm_tags:
                    continue
                local = os.path.join(self.local_dir, name)
                if not os.path.exists(local):
                    # A worker killed mid-download must never leave a
                    # truncated file at the final path: the exists() check
                    # would trust it forever and the crc sweep would push it
                    # over the good GCS copy, poisoning the whole fleet.
                    tmp = local + ".part"
                    blob.download_to_filename(
                        tmp, timeout=self._timeout, retry=retry_policy
                    )
                    os.replace(tmp, local)
                # record what GCS holds; if the local file differs (e.g. it
                # predates this download attempt), the sweep re-pushes it
                self._synced[name] = blob.crc32c
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
        # Sweep the handful of cache files, uploading whatever GCS doesn't
        # already hold. Dirtiness is content identity (crc32c, ~0.2s per
        # 100MB), never mtime — ORT touches every file on every load. The
        # crc is hashed before the upload, so a file modified mid-upload
        # (e.g. the timing cache during a concurrent engine build) looks
        # dirty again on the next sweep and is re-pushed clean. Per-file
        # try: one failed upload must not abandon the rest of the sweep.
        for name in os.listdir(self.local_dir):
            local = os.path.join(self.local_dir, name)
            # .part files are in-flight downloads (see download()), not
            # cache content — never push one to GCS
            if not os.path.isfile(local) or name.endswith(".part"):
                continue
            try:
                crc = _crc32c_b64(local)
                if self._synced.get(name) == crc:
                    continue
                self._gs.upload_file(
                    local,
                    f"{self.prefix}/{name}",
                    timeout=self._timeout,
                    retry=get_gcs_retry_policy(timeout=self._timeout),
                    chunk_size=self._UPLOAD_CHUNK_BYTES,
                )
                self._synced[name] = crc
            except Exception as e:  # noqa: BLE001 - cache is best-effort
                logging.warning(
                    f"TRT cache upload of {name} to {self.prefix} failed: {e}"
                )


# Substrings ORT's TensorRT EP uses when a cached engine/timing file can't
# be loaded (vs. a genuine build failure, which mentions neither). Matched
# case-insensitively; a false positive only costs one extra build attempt.
_TRT_CACHE_LOAD_ERROR_MARKERS = (
    "deserialize engine from cache",
    "engine cache",
    "timing cache",
)


def _is_trt_cache_load_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _TRT_CACHE_LOAD_ERROR_MARKERS)


def _clear_trt_cache_dir(path: str):
    """Delete the files of one scope's cache dir (engines, profiles, timing)."""
    for name in os.listdir(path):
        target = os.path.join(path, name)
        if os.path.isfile(target):
            try:
                os.remove(target)
            except OSError as e:
                logging.warning(f"Could not remove corrupt cache file {target}: {e}")


def create_session(
    model_path: str,
    device: str,
    provider_options: dict | None = None,
    graph_optimizations: bool = True,
):
    """Create an ORT InferenceSession for `device`, failing loudly on fallback.

    `graph_optimizations=False` runs the graph as exported (ORT's
    ORT_DISABLE_ALL). See ImagePredictor for when that is worth doing.

    Returns (session, trt_cache_sync); trt_cache_sync is a _TrtCacheGcsSync
    when device is a tensorrt variant and ORIENT_EXPRESS_TRT_CACHE_GCS is
    set, else None.
    """
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if graph_optimizations
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session_options.enable_mem_pattern = True
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_reuse = True

    trt_scope = None
    if device in TENSORRT_DEVICES:
        trt_scope = trt_cache_scope(
            model_path, provider_options, _TRT_PRECISION[device]
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

    try:
        session = ort.InferenceSession(
            model_path, providers=providers, sess_options=session_options
        )
    except Exception as e:
        # A corrupt cached engine (torn write, bad byte on disk) fails the
        # deserialize here — ORT never falls back to rebuilding on its own.
        # Clear the scope's cache and retry once as a native build; the
        # sweep below then replaces any corrupt GCS copy with the fresh
        # engine. A second failure is a real error and propagates.
        if trt_scope is None or not _is_trt_cache_load_error(e):
            raise
        cache_dir = providers[0][1]["trt_engine_cache_path"]
        logging.warning(
            f"TensorRT cache under {cache_dir} failed to load ({e}); "
            "clearing it and rebuilding the engine natively."
        )
        _clear_trt_cache_dir(cache_dir)
        session = ort.InferenceSession(
            model_path, providers=providers, sess_options=session_options
        )

    # With an explicit profile (mandatory for TRT) the engine builds during
    # session init, not first predict — push it now so even a worker that
    # dies before predicting still populates the shared cache. A cache-hit
    # load makes this a no-op sweep (crc ledger).
    if trt_cache_sync is not None:
        trt_cache_sync.schedule_upload()

    # ORT silently falls back to the CPU EP when a GPU provider can't load
    # (missing CUDA libs, wrong wheel) — in production that is a 10-50x
    # slowdown that looks like a working deployment. Fail loudly.
    if device != Device.CPU:
        active = session.get_providers()[0]
        wanted = DEVICE_TO_PROVIDER[device]
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
