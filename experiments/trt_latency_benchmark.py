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
  - The dynamo export lowers the head's as_strided to sequence ops that
    TensorRT rejects; a sanitize pass rewrites that (verified bit-identical)
    and all TRT paths run the sanitized graph. ort-cuda runs the original
    (production) graph; ort-cuda-sanitized isolates the graph-cleanup effect.
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

MODEL_NAME_CANDIDATES = ["dg-otc-feature-extraction-small"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx-path", help="skip the model registry, use this onnx file")
    p.add_argument("--model-name", default=None, help="registry display name")
    p.add_argument("--project", default="shiftsmart-api")
    p.add_argument("--region", default="us-west1")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument(
        "--opt-batch",
        type=int,
        default=None,
        help="profile opt batch for both TRT paths (default: max batch)",
    )
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--precision",
        choices=["fp32", "fp16", "both"],
        default="both",
        help="precision variants for the two TensorRT paths",
    )
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
    print(
        f"downloading {vertex_model.model_name} v{vertex_model.version} -> {model_dir}"
    )
    vertex_model.download_artifacts(model_dir, force_download=False)
    with open(get_metadata_path(model_dir)) as f:
        metadata = yaml.safe_load(f)
    return os.path.join(model_dir, metadata["model_file"])


def cache_dir_for(onnx_path: str) -> str:
    d = os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "trt_bench_cache")
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------- TRT sanitizer
#
# Two rewrite patterns, both verified bit-identical before use:
#
# 1. The dynamo export of timm's AdaptiveAvgPool2d head lowers `as_strided`
#    to SequenceEmpty -> Loop -> Gather over a flattened copy of the pooled
#    tensor. TensorRT has no sequence ops, so both TRT paths reject the
#    graph. For this model the as_strided is a contiguous identity view: the
#    Gather re-reads its source in order, so consumers can take the
#    pre-flatten tensor directly and the sequence machinery dead-codes away.
#
# 2. Detection exports do input(uint8) -> Transpose -> Cast(float), and
#    TensorRT forbids uint8 intermediate tensors (the ORT TRT EP quietly
#    leaves the Transpose on the CUDA EP instead). Hoisting the Cast above
#    the Transpose removes the uint8 intermediate.


def sanitize_for_trt(onnx_path: str, res: int) -> str:
    import onnx

    model = onnx.load(onnx_path)
    graph = model.graph
    nodes = list(graph.node)
    producer = {out: n for n in nodes for out in n.output}

    def traces_to_loop(name, hops=4):
        node = producer.get(name)
        for _ in range(hops):
            if node is None:
                return False
            if node.op_type == "Loop":
                return True
            node = producer.get(node.input[0]) if node.input else None
        return False

    rewired = 0
    for n in nodes:
        if n.op_type != "Gather" or len(n.input) < 2:
            continue
        data_producer = producer.get(n.input[0])
        if data_producer is None or data_producer.op_type != "Reshape":
            continue
        if not traces_to_loop(n.input[1]):
            continue
        source = data_producer.input[0]
        for consumer in nodes:
            for i, inp in enumerate(consumer.input):
                if inp == n.output[0]:
                    consumer.input[i] = source
                    rewired += 1

    hoisted = 0
    uint8_inputs = {
        i.name
        for i in graph.input
        if i.type.tensor_type.elem_type == onnx.TensorProto.UINT8
    }
    for n in nodes:
        if n.op_type != "Transpose" or n.input[0] not in uint8_inputs:
            continue
        consumers = [c for c in nodes if n.output[0] in c.input]
        if len(consumers) != 1 or consumers[0].op_type != "Cast":
            continue
        cast = consumers[0]
        # images -> Cast -> Transpose; Transpose takes over the Cast's output
        # name so downstream consumers are untouched
        cast_out = cast.output[0]
        cast.input[0] = n.input[0]
        cast.output[0] = cast_out + "_hoisted"
        n.input[0] = cast.output[0]
        old_transpose_out = n.output[0]
        n.output[0] = cast_out
        for c in nodes:
            for i, inp in enumerate(c.input):
                if inp == old_transpose_out and c is not cast:
                    c.input[i] = cast_out
        ti, ci = nodes.index(n), nodes.index(cast)
        if ci > ti:
            nodes[ti], nodes[ci] = nodes[ci], nodes[ti]
        hoisted += 1

    if not (rewired or hoisted):
        return onnx_path
    del graph.value_info[:]  # shape annotations are stale after rewiring

    needed = {o.name for o in graph.output}
    keep = []
    for n in reversed(nodes):
        if any(o in needed for o in n.output):
            keep.append(n)
            needed.update(i for i in n.input if i)
    keep.reverse()
    del graph.node[:]
    graph.node.extend(keep)
    used = {i for n in keep for i in n.input}
    inits = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)
    onnx.checker.check_model(model)

    out_path = os.path.join(cache_dir_for(onnx_path), "model_trt_sanitized.onnx")
    onnx.save(model, out_path)

    import onnxruntime as ort

    orig = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    patched = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    img = find_image_input(orig.get_inputs())
    n = img.shape[0] if isinstance(img.shape[0], int) else 2
    x = np.random.default_rng(7).integers(
        0, 256, size=(n, res, res, 3), dtype=np.uint8
    )
    feed = {img.name: x, **extra_feeds_for(orig.get_inputs(), img.name, res)}
    y0 = orig.run(None, feed)
    y1 = patched.run(None, feed)
    for a, b in zip(y0, y1):
        if not np.array_equal(a, b):
            raise RuntimeError(
                "sanitized model diverges from original, refusing to benchmark it"
            )
    print(
        f"sanitized for TRT: removed {len(nodes) - len(keep)} sequence-op nodes, "
        f"hoisted {hoisted} uint8 casts, outputs bit-identical -> {out_path}"
    )
    return out_path


# ------------------------------------------------------------- model probing


ORT_TO_NP = {
    "tensor(uint8)": np.uint8,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def find_image_input(session_inputs):
    img = next((i for i in session_inputs if len(i.shape) == 4), None)
    if img is None or img.shape[3] != 3:
        raise SystemExit(
            f"no NHWC image input found in {[(i.name, i.shape) for i in session_inputs]}"
        )
    return img


def extra_feeds_for(session_inputs, image_input_name, res):
    """Constant feeds for non-image inputs. Detection models take a
    target_sizes tensor; filling with the input resolution keeps boxes in
    pixel space. Anything fancier needs a model-specific hook."""
    feeds = {}
    for i in session_inputs:
        if i.name == image_input_name:
            continue
        shape = [1 if not isinstance(d, int) else d for d in i.shape]
        feeds[i.name] = np.full(shape, res, dtype=ORT_TO_NP[i.type])
    return feeds


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
    def __init__(self, onnx_path, providers, name, extra_feed=None):
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
        self.input_name = find_image_input(self.session.get_inputs()).name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.extra_feed = extra_feed or {}

    def __call__(self, batch):
        feed = {self.input_name: batch}
        for k, v in self.extra_feed.items():
            if v.shape[0] != len(batch):  # dynamic-batch models want [n, ...]
                v = np.repeat(v[:1], len(batch), axis=0)
            feed[k] = v
        outs = self.session.run(None, feed)
        return dict(zip(self.output_names, outs))


def make_ort_cuda(onnx_path, extra_feed):
    return OrtRunner(onnx_path, ["CUDAExecutionProvider"], "ort-cuda", extra_feed)


def make_ort_trt(
    onnx_path,
    profile_shapes,
    opt_batch,
    max_batch,
    fp16,
    workspace_gb,
    extra_feed,
):
    cache = cache_dir_for(onnx_path)
    opts = {
        "trt_fp16_enable": fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache,
        "trt_max_workspace_size": workspace_gb << 30,
    }
    if profile_shapes:
        # every dynamic-batch input needs a profile, not just the image
        def shapes(batch):
            return ",".join(
                f"{name}:{'x'.join(str(d) for d in (batch, *tail))}"
                for name, tail in profile_shapes.items()
            )

        opts.update(
            {
                "trt_profile_min_shapes": shapes(1),
                "trt_profile_opt_shapes": shapes(opt_batch),
                "trt_profile_max_shapes": shapes(max_batch),
            }
        )
    name = f"ort-trt-{'fp16' if fp16 else 'fp32'}"
    # CUDA EP fallback stays in the list (production would too); if TRT can't
    # take the whole graph the numbers will show it.
    return OrtRunner(
        onnx_path,
        [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"],
        name,
        extra_feed,
    )


# --------------------------------------------------------- native TRT runner


class TrtNativeRunner:
    def __init__(
        self, onnx_path, image_input, opt_batch, max_batch, fp16, workspace_gb,
        extra_inputs=None,
    ):
        import tensorrt as trt

        try:
            from cuda.bindings import runtime as cudart  # cuda-python >= 13
        except ImportError:
            from cuda import cudart  # cuda-python 12.x

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
            shape = [
                max_batch if d == -1 else d for d in self.engine.get_tensor_shape(tname)
            ]
            nbytes = int(np.prod(shape)) * dtype.itemsize
            err, dptr = cudart.cudaMalloc(nbytes)
            self._check(err)
            self.io.append((tname, is_input, dtype, dptr))
            self.context.set_tensor_address(tname, dptr)

        inputs = {t[0]: t for t in self.io if t[1]}
        self.input = inputs.pop(image_input)
        self.dynamic = -1 in tuple(self.engine.get_tensor_shape(image_input))
        self.outputs = [t for t in self.io if not t[1]]

        # non-image inputs are constant for this benchmark: copy once now,
        # tiled to max_batch if their batch dim is dynamic
        extra_inputs = extra_inputs or {}
        self.extra_dynamic = []
        for tname, t in inputs.items():
            arr = np.ascontiguousarray(extra_inputs[tname], dtype=t[2])
            eshape = tuple(self.engine.get_tensor_shape(tname))
            if -1 in eshape:
                reps = [max_batch if d == -1 else 1 for d in eshape]
                arr = np.ascontiguousarray(np.tile(arr, reps))
                self.extra_dynamic.append((tname, tuple(arr.shape[1:])))
            err = cudart.cudaMemcpy(
                t[3],
                arr.ctypes.data,
                arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )[0]
            self._check(err)

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

        # profile only needed for dynamic shapes; assumes dim 0 is the batch
        profile = builder.create_optimization_profile()
        needs_profile = False
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            dims = list(inp.shape)  # e.g. (-1, res, res, 3)
            if -1 not in dims:
                continue
            needs_profile = True
            profile.set_shape(
                inp.name,
                tuple([1] + dims[1:]),
                tuple([opt_batch] + dims[1:]),
                tuple([self.max_batch] + dims[1:]),
            )
        if needs_profile:
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
        if self.dynamic:
            self.context.set_input_shape(name, batch.shape)
            for ename, tail in self.extra_dynamic:
                self.context.set_input_shape(ename, (n,) + tail)

        err = cudart.cudaMemcpyAsync(
            dptr,
            batch.ctypes.data,
            batch.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )[0]
        self._check(err)

        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("execute_async_v3 failed")

        results = {}
        for oname, _, odtype, odptr in self.outputs:
            oshape = tuple(self.context.get_tensor_shape(oname))
            oshape = tuple(n if d == -1 else d for d in oshape)
            host = np.empty(oshape, dtype=odtype)
            err = cudart.cudaMemcpyAsync(
                host.ctypes.data,
                odptr,
                host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )[0]
            self._check(err)
            results[oname] = host

        self._check(cudart.cudaStreamSynchronize(self.stream)[0])
        return results


# --------------------------------------------------------------------- main


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def iou_matrix(a, b):
    """a [N,4], b [M,4] xyxy -> [N,M] IoU."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def detection_agreement(out, ref, top_k=20, iou_thr=0.5):
    """Raw boxes/scores tensors can't be compared elementwise: near-tied
    scores reorder the query slots. Greedy-match the top-k detections per
    image by IoU instead (top-k, not a score threshold, so the check still
    bites on the random benchmark input where nothing is confident)."""
    n_ref = n_match = 0
    ious, score_diffs = [], []
    for i in range(ref["scores"].shape[0]):
        keep_r = np.argsort(ref["scores"][i])[::-1][:top_k]
        keep_o = np.argsort(out["scores"][i])[::-1][:top_k]
        br, bo = ref["boxes"][i][keep_r], out["boxes"][i][keep_o]
        lr, lo = ref["labels"][i][keep_r], out["labels"][i][keep_o]
        sr, so = ref["scores"][i][keep_r], out["scores"][i][keep_o]
        n_ref += len(br)
        m = iou_matrix(br, bo)
        while m.size and m.max() >= iou_thr:
            r, o = np.unravel_index(m.argmax(), m.shape)
            if lr[r] == lo[o]:
                n_match += 1
                ious.append(float(m[r, o]))
                score_diffs.append(abs(float(sr[r]) - float(so[o])))
            m[r, :] = 0
            m[:, o] = 0
    return {
        "top_k": top_k,
        "iou_thr": iou_thr,
        "ref_detections": n_ref,
        "matched": n_match,
        "mean_matched_iou": round(float(np.mean(ious)), 4) if ious else None,
        "max_score_diff": round(max(score_diffs), 4) if score_diffs else None,
    }


def compare_outputs(entry, out, ref):
    if {"boxes", "scores", "labels"} <= set(ref):
        entry["detection_agreement_vs_ort_cuda"] = detection_agreement(out, ref)
        return
    a = next(iter(out.values()))
    b = next(iter(ref.values()))
    entry["cosine_vs_ort_cuda"] = round(cosine(a, b), 6)
    entry["max_abs_diff_vs_ort_cuda"] = float(
        np.abs(a.astype(np.float32) - b.astype(np.float32)).max()
    )


def main():
    args = parse_args()
    onnx_path = resolve_onnx_path(args)
    print(f"model: {onnx_path}")

    import onnxruntime as ort

    probe = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = probe.get_inputs()
    image_input = find_image_input(inputs)
    input_name = image_input.name
    res = image_input.shape[1]
    batch_dim = image_input.shape[0]
    dynamic_batch = not isinstance(batch_dim, int)
    for i in inputs:
        print(f"input: {i.name} {i.shape} {i.type}")
    extra_feed = extra_feeds_for(inputs, input_name, res)
    for k, v in extra_feed.items():
        print(f"constant feed for {k}: {v.tolist()}")
    profile_shapes = {}
    if dynamic_batch:
        profile_shapes = {
            i.name: tuple(i.shape[1:])
            for i in inputs
            if not isinstance(i.shape[0], int)
        }
    del probe

    batch_sizes = sorted(args.batch_sizes)
    if not dynamic_batch and batch_sizes != [batch_dim]:
        print(f"static batch dim {batch_dim}: overriding --batch-sizes")
        batch_sizes = [batch_dim]
    max_batch = batch_sizes[-1]
    opt_batch = args.opt_batch or max_batch
    trt_precisions = (
        [False, True] if args.precision == "both" else [args.precision == "fp16"]
    )

    trt_onnx_path = sanitize_for_trt(onnx_path, res)

    rng = np.random.default_rng(0)
    batches = {
        n: rng.integers(0, 256, size=(n, res, res, 3), dtype=np.uint8)
        for n in batch_sizes
    }
    ref_batch = batches[batch_sizes[0]]

    runner_makers = [("ort-cuda", lambda: make_ort_cuda(onnx_path, extra_feed))]
    if trt_onnx_path != onnx_path:
        # control: how much of any TRT win is just the graph cleanup?
        runner_makers.append(
            ("ort-cuda-sanitized", lambda: make_ort_cuda(trt_onnx_path, extra_feed))
        )
    for fp16 in trt_precisions:
        runner_makers.append(
            (
                f"ort-trt-{'fp16' if fp16 else 'fp32'}",
                lambda fp16=fp16: make_ort_trt(
                    trt_onnx_path,
                    profile_shapes,
                    opt_batch,
                    max_batch,
                    fp16,
                    args.workspace_gb,
                    extra_feed,
                ),
            )
        )
        runner_makers.append(
            (
                f"trt-{'fp16' if fp16 else 'fp32'}",
                lambda fp16=fp16: TrtNativeRunner(
                    trt_onnx_path,
                    input_name,
                    opt_batch,
                    max_batch,
                    fp16,
                    args.workspace_gb,
                    extra_feed,
                ),
            )
        )

    results = {
        "onnx_path": onnx_path,
        "trt_onnx_path": trt_onnx_path,
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
            compare_outputs(entry, out, reference_out)

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
        elif "detection_agreement_vs_ort_cuda" in entry:
            d = entry["detection_agreement_vs_ort_cuda"]
            print(
                f"  agreement vs ort-cuda: {d['matched']}/{d['ref_detections']} "
                f"top-{d['top_k']} detections matched (IoU>={d['iou_thr']}, "
                f"mean IoU {d['mean_matched_iou']}, "
                f"max score diff {d['max_score_diff']})"
            )
        results["backends"][name] = entry
        del runner

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    # relative summary
    ok = {n: e for n, e in results["backends"].items() if "batches" in e}
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
