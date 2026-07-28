# Model requirements

What an exported model must look like to work with this library, and which
optional properties unlock which functionality. Written for whoever exports
the next model (see `ml-models`); every hard requirement here is enforced
with an explicit error at load time.

## Artifact layout

A model is a directory (uploaded to the Vertex registry as one version)
containing:

- `metadata.yaml` with:
  - `model_type` — selects the predictor class (`object-detection-onnx`,
    `feature-extraction-onnx`, `classification-onnx`,
    `multi-label-classification-onnx`, `semantic-segmentation-onnx`,
    `instance-segmentation-onnx`, or `joblib`). Strings are unique and
    persisted in uploaded metadata — never reuse one for a different
    output contract.
  - `model_file` — the ONNX file's name.
  - `classes` — `{int: str}` label map (image predictors).
- the ONNX model, self-contained in a single file (no external data files).

`joblib` artifacts (tabular models) are exempt from everything below.

## Input contract (hard requirements)

- The **first input is the image batch**: `[batch, H, W, 3]` — **NHWC,
  uint8**. Loading refuses channels-first (NCHW) exports.
- `H == W` (the predictor derives its resolution from dim 1 and resizes
  squares).
- Detection / instance segmentation additionally take
  `target_sizes: float32 [batch, 2]` (original height, width) as the
  second input.
- Output shapes/dtypes must match the predictor type's `postprocess`
  (e.g. detection: `boxes [batch, N, 4]`, `scores [batch, N]`,
  `labels [batch, N]`). The golden equivalence suite is the contract test —
  add a case for every new model.

## Batch dimension (decides most functionality)

Export with a **dynamic batch dimension** (`[batch, ...]`, not `[1, ...]`).
A static batch-1 export loads, but:

- `predict()` works only one image at a time — any batch > 1 is an ORT
  shape error;
- `predict_stream` / `ImageLoader` batching and the throughput gains that
  motivate them are unavailable;
- TensorRT profiles degenerate to min=opt=max at batch 1.

Static exports are considered out of compliance for production use
(`dg-otc-detection` v1 is the standing example).

## TensorRT (`device="tensorrt"` / `"tensorrt-fp16"`)

Requirements beyond the above, roughly in the order they will bite:

1. **An explicit optimization profile is mandatory.** Construction raises
   unless `provider_options` carries all three of
   `trt_profile_min/opt/max_shapes` covering every shape you will send
   (ragged final batches included). Fixed-shape models declare
   min = opt = max.
2. **The graph must survive ORT's TRT capability pass.** Internal tensors
   need resolvable shapes — a graph that fails with `TensorRT input: X has
   no shape specified` cannot load on TRT at all (the current
   feature-extraction export fails this; ORT's `symbolic_shape_infer` also
   chokes on it, so only a cleaner re-export fixes it).
3. **uint8 image inputs get partitioned.** TRT rejects the uint8 subgraph,
   ORT runs it on CUDA, and two consequences follow: red (harmless) ERROR
   log spam on every load, and profile specs must ALSO name the internal
   partition-boundary tensor (e.g. `/Transpose_output_0:16x3x576x576`) —
   ORT names it in its session-init error. A TRT-clean export — one that
   applies the uint8→float cast before the NHWC→NCHW transpose, so no uint8
   tensor reaches the TRT partition boundary — avoids both.
4. **fp16 needs per-model accuracy validation on real photos before use.**
   The failure mode is not gradual drift: a graph that doesn't tolerate
   fp16 can return scores crushed below production thresholds (an older
   detection export returned 0 detections at fp16 while newer exports of
   the same model are healthy). Validate against CPU outputs per model,
   per export.
5. TRT fp32 is not bit-comparable to CPU/CUDA — expect small score drift;
   validate models with tight geometric requirements (an older
   instance-segmentation export shifts boxes up to ~10px on TRT fp32).

Engine caches are scoped by model bytes, ORT/TRT versions, profile,
precision, other engine-affecting provider options, and
`NVIDIA_TF32_OVERRIDE` — a re-export or version bump automatically gets a
fresh scope (see README, "TensorRT Engine Caching").

## Upload discipline

- **Never overwrite a registry version in place.** Local artifact caches
  are keyed `{model_name}-{version}` and downloaded with
  `force_download=False`, so an overwritten version keeps serving stale
  bytes on every machine that ever downloaded it. Changed artifact ⇒ new
  version.
- After uploading, add/refresh the model's golden equivalence case
  (`tests/equivalence/`) and review the report before relying on it.
