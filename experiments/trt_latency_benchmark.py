"""Latency experiment: ORT CUDA EP vs ORT TensorRT EP vs native TensorRT.

Question this answers: is the ORT TensorRT execution provider close enough to
native TensorRT that a native-TRT backend in orient_express is unnecessary?

Backends benchmarked (all host-numpy in -> host-numpy out, which is how
orient_express predictors call onnxruntime today):
  - ort-cuda            CUDAExecutionProvider, fp32 (production baseline)
  - ort-trt-fp32/fp16   TensorrtExecutionProvider, engine+timing cache on,
                        explicit optimization profile
  - trt-fp32/fp16       native TensorRT engine (onnx -> plan), executed with
                        cuda-python; H2D copy + execute_async_v3 + D2H + sync
                        inside the timed region

Fairness notes:
  - Both TRT paths share the same optimization profile
    (min=1, opt=--opt-batch, max=max of --batch-sizes).
  - Native TRT copies use pageable numpy memory, same as ORT's run(); a
    tuned native deployment (pinned memory, CUDA graphs) could shave a bit
    more, so treat the native-TRT numbers as a conservative floor.
  - First ORT-TRT run and native engine build are timed separately and
    excluded from latency stats; engines are cached on disk, so re-runs of
    the script skip the build.
  - A fixed batch is pushed through every backend and compared to ort-cuda
    (cosine + max abs diff) so a "fast" backend that silently degrades the
    embeddings is visible.

Setup on the 5090 box (Blackwell / SM 12.0 needs recent everything):
    pip install "onnxruntime-gpu>=1.22" tensorrt cuda-python numpy pyyaml
  - onnxruntime-gpu must be a CUDA 12.x build, and the pip `tensorrt` major
    version must match what ORT was built against (ORT prints a clear error
    on mismatch and this script will report the EP as unavailable).
  - If the TRT EP fails to load libnvinfer, the usual fix is
    LD_LIBRARY_PATH=$(python -c 'import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))')

Usage:
    python experiments/trt_latency_benchmark.py \
        --project shiftsmart-api --region us-west1 \
        --batch-sizes 1 8 32 --iters 100 --warmup 20

    # or skip the registry and point at a local onnx file:
    python experiments/trt_latency_benchmark.py --onnx-path model.onnx
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME_CANDIDATES = ["dg-otc-feature-extractor", "dg_otc_feature_extractor"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx-path", help="skip the model registry, use this onnx file")
    p.add_argument("--model-name", default=None, help="registry display name")
    p.add_argument("--project", default="shiftsmart-api")
    p.add_argument("--region", default="us-west1")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--opt-batch", type=int, default=None,
                   help="profile opt batch for both TRT paths (default: max batch)")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--precision", choices=["fp32", "fp16", "both"], default="both",
                   help="precision variants for the two TensorRT paths")
    p.add_argument("--workspace-gb", type=int, default=4)
    p.add_argument("--out", default="trt_benchmark_results.json")
    return p.parse_args()


def resolve_onnx_path(args) -> str:
    if args.onnx_path:
        return args.onnx_path

    import yaml

    from orient_express.utils.paths import get_cache_dir, get_metadata_path
    from orient_express.vertex import get_vertex_model

    candidates = [args.model_name] if args.model_name else MODEL_NAME_CANDIDATES
    vertex_model = None
    for name in candidates:
        vertex_model = get_vertex_model(
            name, args.project, args.region, raise_exception=False
        )
        if vertex_model is not None:
            break
    if vertex_model is None:
        raise SystemExit(f"model not found in registry under any of: {candidates}")

    model_dir = os.path.join(
        get_cache_dir(), f"{vertex_model.model_name}-{vertex_model.version}"
    )
    print(f"downloading {vertex_model.model_name} v{vertex_model.version} -> {model_dir}")
    vertex_model.download_artifacts(model_dir, force_download=False)
    with open(get_metadata_path(model_dir)) as f:
        metadata = yaml.safe_load(f)
    return os.path.join(model_dir, metadata["model_file"])


def cache_dir_for(onnx_path: str) -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "trt_bench_cache")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- benchmark


def bench(run, batch, iters, warmup):
    for _ in range(warmup):
        run(batch)
    times = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter()
        run(batch)
        times[i] = time.perf_counter() - t0
    times *= 1000.0
    return {
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "p99_ms": float(np.percentile(times, 99)),
        "imgs_per_s": float(len(batch) / (times.mean() / 1000.0)),
    }


# --------------------------------------------------------------- ORT runners


class OrtRunner:
    def __init__(self, onnx_path, providers, name):
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=so
        )
        self.session_create_s = time.perf_counter() - t0
        self.name = name
        active = self.session.get_providers()
        wanted = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if active[0] != wanted:
            raise RuntimeError(f"{name}: wanted {wanted}, session picked {active}")
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, batch):
        return self.session.run(None, {self.input_name: batch})[0]


def make_ort_cuda(onnx_path):
    return OrtRunner(onnx_path, ["CUDAExecutionProvider"], "ort-cuda")


def make_ort_trt(onnx_path, input_name, res, opt_batch, max_batch, fp16, workspace_gb):
    cache = cache_dir_for(onnx_path)
    opts = {
        "trt_fp16_enable": fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache,
        "trt_max_workspace_size": workspace_gb << 30,
        "trt_profile_min_shapes": f"{input_name}:1x{res}x{res}x3",
        "trt_profile_opt_shapes": f"{input_name}:{opt_batch}x{res}x{res}x3",
        "trt_profile_max_shapes": f"{input_name}:{max_batch}x{res}x{res}x3",
    }
    name = f"ort-trt-{'fp16' if fp16 else 'fp32'}"
    # CUDA EP fallback stays in the list (production would too); if TRT can't
    # take the whole graph the numbers will show it.
    return OrtRunner(
        onnx_path,
        [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"],
        name,
    )


# --------------------------------------------------------- native TRT runner


class TrtNativeRunner:
    def __init__(self, onnx_path, opt_batch, max_batch, fp16, workspace_gb):
        import tensorrt as trt
        from cuda import cudart

        self.trt = trt
        self.cudart = cudart
        self.name = f"trt-{'fp16' if fp16 else 'fp32'}"
        self.max_batch = max_batch

        logger = trt.Logger(trt.Logger.WARNING)
        plan_path = os.path.join(
            cache_dir_for(onnx_path), f"native_{'fp16' if fp16 else 'fp32'}.plan"
        )
        self.build_s = 0.0
        if os.path.exists(plan_path):
            plan = open(plan_path, "rb").read()
        else:
            t0 = time.perf_counter()
            plan = self._build(onnx_path, opt_batch, fp16, workspace_gb, logger)
            self.build_s = time.perf_counter() - t0
            with open(plan_path, "wb") as f:
                f.write(plan)

        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        self._check(err)

        self.io = []  # (name, is_input, np_dtype, device_ptr)
        for i in range(self.engine.num_io_tensors):
            tname = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(tname) == trt.TensorIOMode.INPUT
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(tname)))
            shape = [max_batch if d == -1 else d for d in self.engine.get_tensor_shape(tname)]
            nbytes = int(np.prod(shape)) * dtype.itemsize
            err, dptr = cudart.cudaMalloc(nbytes)
            self._check(err)
            self.io.append((tname, is_input, dtype, dptr))
            self.context.set_tensor_address(tname, dptr)

        inputs = [t for t in self.io if t[1]]
        if len(inputs) != 1:
            raise RuntimeError(f"expected 1 input tensor, engine has {len(inputs)}")
        self.input = inputs[0]
        self.outputs = [t for t in self.io if not t[1]]

    def _build(self, onnx_path, opt_batch, fp16, workspace_gb, logger):
        trt = self.trt
        builder = trt.Builder(logger)
        flags = 0
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("onnx parse failed:\n" + "\n".join(errs))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        inp = network.get_input(0)
        dims = list(inp.shape)  # (-1, res, res, 3)
        profile.set_shape(
            inp.name,
            tuple([1] + dims[1:]),
            tuple([opt_batch] + dims[1:]),
            tuple([self.max_batch] + dims[1:]),
        )
        config.add_optimization_profile(profile)

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")
        return bytes(plan)

    def _check(self, err):
        if int(err) != 0:
            raise RuntimeError(f"CUDA error: {self.cudart.cudaGetErrorString(err)}")

    def __call__(self, batch):
        cudart = self.cudart
        n = batch.shape[0]
        name, _, dtype, dptr = self.input
        batch = np.ascontiguousarray(batch, dtype=dtype)
        self.context.set_input_shape(name, batch.shape)

        err = cudart.cudaMemcpyAsync(
            dptr, batch.ctypes.data, batch.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        )[0]
        self._check(err)

        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")

        results = []
        for oname, _, odtype, odptr in self.outputs:
            oshape = tuple(self.context.get_tensor_shape(oname))
            oshape = tuple(n if d == -1 else d for d in oshape)
            host = np.empty(oshape, dtype=odtype)
            err = cudart.cudaMemcpyAsync(
                host.ctypes.data, odptr, host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
            )[0]
            self._check(err)
            results.append(host)

        self._check(cudart.cudaStreamSynchronize(self.stream)[0])
        return results[0]


# --------------------------------------------------------------------- main


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    args = parse_args()
    onnx_path = resolve_onnx_path(args)
    print(f"model: {onnx_path}")

    import onnxruntime as ort

    probe = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = probe.get_inputs()
    if len(inputs) != 1:
        raise SystemExit(f"expected a single-input model, got {[i.name for i in inputs]}")
    input_name = inputs[0].name
    res = inputs[0].shape[1]
    print(f"input: {input_name} {inputs[0].shape} {inputs[0].type}")
    del probe

    batch_sizes = sorted(args.batch_sizes)
    max_batch = batch_sizes[-1]
    opt_batch = args.opt_batch or max_batch
    trt_precisions = (
        [False, True] if args.precision == "both" else [args.precision == "fp16"]
    )

    rng = np.random.default_rng(0)
    batches = {
        n: rng.integers(0, 256, size=(n, res, res, 3), dtype=np.uint8)
        for n in batch_sizes
    }
    ref_batch = batches[batch_sizes[0]]

    runner_makers = [("ort-cuda", lambda: make_ort_cuda(onnx_path))]
    for fp16 in trt_precisions:
        runner_makers.append((
            f"ort-trt-{'fp16' if fp16 else 'fp32'}",
            lambda fp16=fp16: make_ort_trt(
                onnx_path, input_name, res, opt_batch, max_batch, fp16, args.workspace_gb
            ),
        ))
        runner_makers.append((
            f"trt-{'fp16' if fp16 else 'fp32'}",
            lambda fp16=fp16: TrtNativeRunner(
                onnx_path, opt_batch, max_batch, fp16, args.workspace_gb
            ),
        ))

    results = {
        "onnx_path": onnx_path,
        "batch_sizes": batch_sizes,
        "opt_batch": opt_batch,
        "iters": args.iters,
        "warmup": args.warmup,
        "backends": {},
    }
    reference_out = None

    for name, make in runner_makers:
        print(f"\n=== {name} ===")
        try:
            runner = make()
        except Exception as e:
            print(f"  UNAVAILABLE: {e}")
            results["backends"][name] = {"error": str(e)}
            continue

        entry = {"batches": {}}
        if getattr(runner, "session_create_s", 0):
            entry["session_create_s"] = round(runner.session_create_s, 2)
        if getattr(runner, "build_s", 0):
            entry["engine_build_s"] = round(runner.build_s, 2)

        # first call on ORT-TRT triggers the (cached) engine build
        t0 = time.perf_counter()
        out = runner(ref_batch)
        entry["first_run_s"] = round(time.perf_counter() - t0, 2)

        if reference_out is None:
            reference_out = out
        else:
            entry["cosine_vs_ort_cuda"] = round(cosine(out, reference_out), 6)
            entry["max_abs_diff_vs_ort_cuda"] = float(
                np.abs(out.astype(np.float32) - reference_out.astype(np.float32)).max()
            )

        for n in batch_sizes:
            stats = bench(runner, batches[n], args.iters, args.warmup)
            entry["batches"][n] = stats
            print(
                f"  batch {n:>3}: mean {stats['mean_ms']:8.3f} ms  "
                f"p50 {stats['p50_ms']:8.3f}  p90 {stats['p90_ms']:8.3f}  "
                f"p99 {stats['p99_ms']:8.3f}  ({stats['imgs_per_s']:8.1f} img/s)"
            )
        if "cosine_vs_ort_cuda" in entry:
            print(
                f"  agreement vs ort-cuda: cosine {entry['cosine_vs_ort_cuda']}, "
                f"max|diff| {entry['max_abs_diff_vs_ort_cuda']:.4g}"
            )
        results["backends"][name] = entry
        del runner

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    # relative summary
    ok = {
        n: e for n, e in results["backends"].items() if "batches" in e
    }
    if "ort-cuda" in ok and len(ok) > 1:
        print("\nspeedup vs ort-cuda (mean latency):")
        for n in batch_sizes:
            base = ok["ort-cuda"]["batches"][n]["mean_ms"]
            row = "  ".join(
                f"{name} {base / e['batches'][n]['mean_ms']:.2f}x"
                for name, e in ok.items()
                if name != "ort-cuda"
            )
            print(f"  batch {n:>3}: {row}")


if __name__ == "__main__":
    main()
