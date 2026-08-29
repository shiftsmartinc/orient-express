# Orient Express

A library to accelerate model deployments to Vertex AI directly from colab notebooks

![train-resized](https://github.com/user-attachments/assets/f1ed32ec-07d9-4d48-8b96-3323db6b5091)

Orient Express provides two main capabilities:

1. **Vertex Model Deployment and Retrieval**: Capabilities for uploading, downloading, or deploying models to Vertex AI Model Registry.

1. **ONNX Image Model Deployment**: Built-in predictor classes for easily running image classification, object detection, instance segmentation, and semantic segmentation models exported to ONNX format.

Both workflows handle versioning, artifact storage in GCS, and integration with Vertex AI Model Registry.

## Installation

Pick the inference runtime you need (a bare `pip install orient_express`
installs no ONNX runtime — fine for registry/upload-only use):

```bash
pip install 'orient_express[cpu]'       # CPU inference
pip install 'orient_express[cuda]'      # NVIDIA GPU; bundles CUDA/cuDNN wheels (py>=3.11)
pip install 'orient_express[tensorrt]'  # GPU + TensorRT (device="tensorrt"), fastest
```

The GPU extras are Linux x86_64 only. They include the CUDA runtime
wheels, so they work on machines without a system CUDA installation — only
the NVIDIA driver is required. Never install the
`cpu` extra together with a GPU extra: both ship the same `onnxruntime`
import package and the winner is install-order-dependent. uv refuses the
combination outright; with pip it's on you.

The GPU extras above are CUDA-13 builds and need NVIDIA driver r580+. On an
older driver (r525+), use the CUDA-12 stack instead — same features, older
ORT line:

```bash
pip install 'orient_express[cuda12]'      # CUDA EP on driver < r580
pip install 'orient_express[tensorrt12]'  # + TensorRT EP
```

Never combine the cu12 and cu13 extras; their pins conflict on purpose so a
mixed install fails at resolution. If a GPU device fails to load, the error
message reports your driver version and which stack it supports.

For local development (uses [uv](https://docs.astral.sh/uv/)):

```bash
make install   # uv sync
make test      # run the test suite
make fmt       # format + autofix lint (run before committing)
make lint      # check-only, same as CI
```

Model-output equivalence testing (internal golden suite; produces an HTML
before/after report): see `tests/equivalence/README.md` and the
`make equivalence` / `make equivalence-docker` targets.


## Workflows

### ONNX Image Model Workflow

This workflow is for deploying image models (classification, detection, segmentation) exported to ONNX format.

```python
from orient_express.predictors import ClassificationPredictor
from orient_express.vertex import upload_model, get_vertex_model

# 1. Create predictor from your exported ONNX model
predictor = ClassificationPredictor(
    onnx_path="model.onnx",
    classes={1: "cat", 2: "dog", 3: "bird"}
)

# 2. Upload to Vertex AI Model Registry
vertex_model = upload_model(
    model=predictor,
    model_name="my-classifier",
    project_name="my-project",
    region="us-central1",
    bucket_name="my-artifacts-bucket",
)

# 3. Later, retrieve and run locally
vertex_model = get_vertex_model(
    model_name="my-classifier",
    project_name="my-project",
    region="us-central1",
)
local_predictor = vertex_model.get_local_predictor()

from PIL import Image
images = [Image.open("test.jpg")]
predictions = local_predictor.predict(images)

# 4. Or deploy to an endpoint for remote inference
vertex_model.deploy_to_endpoint(
    endpoint_name="my-classifier-endpoint",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=3,
)

# remote prediction API depends on the endpoint container deployed with the model
predictions = vertex_model.remote_predict(
    "my-classifier-endpoint",
    [{"image": "https://storage.googleapis.com/ssm-media-uploads/example.jpg"}],
)
```

### Joblib Model Workflow

This workflow is for deploying models that can be serialized with joblib, such as scikit-learn pipelines or XGBoost models.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import seaborn as sns

from orient_express.vertex import upload_model_joblib, get_vertex_model

# 1. Train your model
data = sns.load_dataset('titanic').dropna(subset=['survived'])
X = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = data['survived']

numeric_features = ['age', 'fare', 'sibsp', 'parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['pclass', 'sex', 'embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

model.fit(X, y)

# 2. Upload to Vertex AI Model Registry
vertex_model = upload_model_joblib(
    model=model,
    model_name="titanic-classifier",
    project_name="my-project",
    region="us-central1",
    bucket_name="my-artifacts-bucket",
    serving_container_image_uri="your-serving-container:latest",
    serving_container_health_route="/health",
    serving_container_predict_route="/predict",
)

# 3. Later, retrieve the model
vertex_model = get_vertex_model(
    model_name="titanic-classifier",
    project_name="my-project",
    region="us-central1",
)

# 4. Run locally
local_predictor = vertex_model.get_local_predictor()
predictions = local_predictor.predict(X_test)
```

## ONNX Runtime and Device Support

What a model export must satisfy to work here (input layout, dynamic
batch, TensorRT-specific constraints): see
[MODEL_REQUIREMENTS.md](MODEL_REQUIREMENTS.md).

### Selecting the Execution Device

When loading a predictor, use the `device` parameter to pick the execution
provider. Requesting a GPU device that can't actually load raises instead of
silently running on CPU.

```python
from orient_express.predictors import BoundingBoxPredictor

# device is a plain string; orient_express.predictors.Device provides the
# same values as constants (Device.CUDA == "cuda") if you prefer them
predictor = BoundingBoxPredictor("/path/to/model", classes, device="cpu")

# CUDA (Linux x64, [cuda] extra). Benchmarked on our RF-DETR
# detector: ~26x over CPU.
predictor = BoundingBoxPredictor("/path/to/model", classes, device="cuda")

# TensorRT ([tensorrt] extra): ~1.6x over CUDA at fp32; "tensorrt-fp16"
# is ~3x over CUDA if the model tolerates fp16 (validate accuracy first).
# TensorRT needs an optimization profile; trt_batch is the whole of it —
# see "TensorRT Optimization Profiles".
predictor = BoundingBoxPredictor(
    "/path/to/model", classes, device="tensorrt", trt_batch=8
)
predictor = BoundingBoxPredictor(
    "/path/to/model", classes, device="tensorrt-fp16", trt_batch=8
)

# "tensorrt-bf16": near-fp16 speed with fp32's dynamic range. The 16-bit
# mode for models whose activations overflow fp16's 65504 max — e.g.
# DINOv3-backbone models, which carry ~1.5e5 residual activations and NaN
# under fp16. Validate accuracy per model, same as fp16.
predictor = BoundingBoxPredictor(
    "/path/to/model", classes, device="tensorrt-bf16", trt_batch=8
)

# same values work when loading from Vertex
predictor = model.get_local_predictor(device="cuda")
```

#### Graph optimizations

ONNX Runtime's graph fusions are enabled by default. Pass
`graph_optimizations=False` (to a predictor constructor or `get_predictor`)
to run the graph exactly as exported. Worth knowing about because the
serving image sets it for CPU and CUDA: that image is pinned to the CUDA-12
onnxruntime line, and 1.24.x mis-fuses DINOv3-backbone graphs badly enough
to return a different answer in every process. Disabling fusion costs
~14-17% on RF-DETR through those providers.

TensorRT keeps optimizations on, and should: the bug is in onnxruntime's own
fused kernels, TensorRT executes the numerics itself, and it re-optimizes
regardless so fusion costs it nothing either way. Some graphs also only
import into TensorRT once onnxruntime's shape inference has run over them —
disabling optimizations makes them fail to load.

### Choosing the Serving Device at Deploy Time

The runtime is a property of the *deployment*, not the model artifact, so
`deploy_to_endpoint` takes it. Nothing about the device is recorded at
upload: every model already in the registry can serve on any device, and one
model version can serve different devices on different endpoints.

```python
model = get_vertex_model("pepsi-food-detection", project_name=..., region=...)

# CPU endpoint
model.deploy_to_endpoint("detect-cpu", "n1-standard-4", 1, 3)

# same model version, L4 endpoint running TensorRT fp16
model.deploy_to_endpoint(
    "detect-gpu", "g2-standard-8", 1, 3,
    accelerator_type="NVIDIA_L4", accelerator_count=1,
    device="tensorrt-fp16",
)
```

Vertex fixes a container's environment at model upload, so the device can't
travel as an env var. It is recorded in an **`oe-device` label on the
endpoint** instead, which the container reads at boot off the endpoint named
by the `AIP_ENDPOINT_ID` variable Vertex injects. For a TensorRT device it
also synthesizes the required optimization profile from the model's own
inputs, covering batches 1..`TRT_MAX_BATCH_SIZE` and splitting larger
requests.

The endpoint carries it, rather than the deployed model, because Vertex
attaches a model to its endpoint only after the container passes its health
checks — a booting container asking for its own deployed model does not find
it, and cannot wait for it either, since the registration it would be
waiting on comes after the health check it would be blocking. The endpoint
already exists when the container starts. A label rather than the display
name, because the name is how you and your dashboards refer to an endpoint
and this library does not rewrite it.

A device therefore belongs to an endpoint and covers every model on it.
Deploying a different device to an endpoint that already serves something
is refused, with the fix in the message: use another endpoint, or undeploy
that one first. An endpoint with nothing deployed yet is simply labelled.

Omitting `device` lets the hardware decide: `cuda` when the deployment has a
GPU attached, else `cpu`. CUDA is numerically identical to CPU on every model
measured, so that default is free — it only stops an accelerator from sitting
idle. The TensorRT tiers change the numbers and are never chosen for you.

A device you *did* name is never second-guessed: it fails the rollout rather
than quietly serving something else, so `tensorrt-fp16` on a model that NaNs
under fp16 (e.g. a DINOv3 backbone) is a failed deployment, not bad
predictions. Validate a device before deploying with it.

Deploying a mismatched pair warns at the call site rather than waiting for
the container to fail: a GPU device on a machine with no accelerator, or
`device="cpu"` on a machine that has one. The first is only a warning
because A2/G2 machine types carry GPUs without an `accelerator_count`.

The container reads the endpoint through the Vertex API, so the service
account it runs as needs `aiplatform.endpoints.get`. Without that permission
the lookup warns and falls back to the hardware default — the deployment
comes up healthy and serves, just not on the device that was asked for. The
identity Vertex assigns by default may not carry that permission, so a
`device=` deployment generally wants an explicit `service_account` (below).

Passing `device` therefore makes `deploy_to_endpoint` check the live
container afterwards and warn if the provider it ended up on isn't the one
requested. The container log names the device it resolved
(`serving device: ...`) and records why if it could not honour the token;
`container_logging` keeps that stream on by default for this reason, and
`container_logging=False` opts out.

Which account that is depends on how the deployment runs. Left unset, the
container runs under an identity Vertex assigns, shared with other custom
containers in the project — so a permission granted for one is granted for
all of them, and it is not guaranteed to include `endpoints.get`.
`service_account` names one per deployment instead:

```python
model.deploy_to_endpoint(
    "detect-gpu", "g2-standard-8", 1, 3,
    accelerator_type="NVIDIA_L4", accelerator_count=1,
    device="tensorrt-fp16",
    service_account="serving@my-project.iam.gserviceaccount.com",
)
```

A user-managed account keeps this deployment's permissions to itself, but it
replaces the default identity entirely, so it needs everything the container
does: read access to the model's artifacts, Logging and Monitoring writes,
and `aiplatform.endpoints.get` when `device` is set.

### TensorRT Engine Caching

TensorRT compiles the model into a GPU-specific engine on first use (minutes
for a mid-size model). Engines and timing caches are stored under the
orient-express cache dir (`ORIENT_EXPRESS_TRT_CACHE_DIR` overrides) and are
reused across processes, so only the first run on a machine pays the build.
Least-recently-used entries are evicted once the local cache exceeds
`ORIENT_EXPRESS_TRT_CACHE_MAX_BYTES` (default 20GB; 0 disables) — an
evicted engine just rebuilds on its next use.

For short-lived workers (e.g. Vertex AI pipelines on a fixed GPU type), set

```bash
ORIENT_EXPRESS_TRT_CACHE_GCS=gs://my-bucket/trt-cache/my-pool
```

and workers download prebuilt engines at startup and upload newly built ones
after inference — each engine build is paid once, org-wide. Cache entries
are scoped automatically by model content hash, ORT and TensorRT versions,
optimization profile, precision, and any other engine-affecting provider
options, so one bucket prefix serves every
model and pool: a worker downloads only the entries for exactly what it
loads. Entries orphaned by model or version bumps are never fetched again —
add a GCS lifecycle rule on the prefix (e.g. delete after 60 days) to
garbage-collect them; an evicted live engine just gets rebuilt and
re-uploaded once.

Uploads run on a background thread and never block inference or process
exit. Sync failures log a warning and degrade to a local build.
`ORIENT_EXPRESS_TRT_CACHE_TIMEOUT` (default 60s) bounds each sync call's
connect/inactivity time AND its retry window, so a GCS outage stalls a
cold start on the order of the timeout per cache file, not the client's
120s retry default. An actively streaming download may exceed it; an
upload currently may not (its whole body shares one deadline).

Measured cold-start cost (RTX 5090, dg-otc detection, 118MB engine,
~2.5MB/s uplink; session-ready time):

| cold engine build | local cache hit | GCS cache hit |
|---|---|---|
| 16.6s | 9.1s | 20.1s |

On a fast-building GPU like this one the GCS hit can cost more than the
build itself — the org-wide cache pays off when builds are slow relative to
the download (bigger models, datacenter GPUs like L4, where builds take
minutes).

### TensorRT Optimization Profiles

Engines are compiled for a shape range (the optimization profile), so one
engine covers every batch size you send. TensorRT devices require the
profile — loading raises without one, because TRT would otherwise compile
for the first shape it sees and any new shape would mean another
multi-minute build.

Every dimension except the batch is pinned by the graph, so `trt_batch` is
normally the whole profile:

```python
# engine covers batches 1..32, tuned for 32
predictor = BoundingBoxPredictor(path, classes, device="tensorrt", trt_batch=32)

# (min, opt, max) when the dominant shape isn't the largest — serving mostly
# gets single images but must tolerate bigger requests
predictor = BoundingBoxPredictor(path, classes, device="tensorrt",
                                 trt_batch=(1, 1, 8))
```

A bare `N` means `(1, N, N)`: min stays 1 so a ragged final batch stays in
profile, and opt tunes for `N`. The rest of each input's shape is read off
the model, and the batch is matched by its ONNX symbolic dim name, so
multi-input models (detection's `images` + `target_sizes`) get consistent
bounds on every input automatically.

Two cases can't be inferred and raise instead: a model with more than one
distinct dynamic dimension (which one is the batch?), and a static export,
which can only ever serve its pinned batch. Both — plus graphs whose TRT
partition has internal dynamic inputs, e.g. `/Transpose_output_0` — take
explicit shapes instead:

```python
predictor = BoundingBoxPredictor(
    path, classes, device="tensorrt",
    provider_options={
        "trt_profile_min_shapes": "images:1x576x576x3,target_sizes:1x2",
        "trt_profile_opt_shapes": "images:32x576x576x3,target_sizes:32x2",
        "trt_profile_max_shapes": "images:32x576x576x3,target_sizes:32x2",
    },
)
```

Pass one or the other, never both. Out-of-profile inputs raise a clear error
instead of being handed to ORT (which would fail the call and silently run
it on CUDA — an invisible performance downgrade in production).

### Streaming and Pipelined Inference

`predict()` is the all-in-one call. Its three stages are also public —
`preprocess` (CPU), `infer` (GPU), `postprocess` (CPU) — and
`predict_stream()` pipelines them over any iterable of image batches,
overlapping data loading and CPU work with GPU inference:

```python
# any iterable of image batches works; a (payload, images) tuple carries
# metadata through to (payload, predictions)
for rows, preds in predictor.predict_stream(my_batches(), confidence=0.4):
    ...
```

Two loaders supply the batches; pick by answering one question — do you
have URLs, or custom loading logic?

**`UrlImageLoader`** — your items map to URLs of encoded images (the
standard case: photos in GCS). The loader owns downloading AND decoding:
downloads run on an asyncio event loop (object-store latency means high
throughput needs hundreds of requests in flight, which threads pay GIL
tax to hold), decoding runs on cv2, which releases the GIL. URLs are
fetched exactly as given — no credentials are attached; pass `headers=`
if the endpoint needs auth.

```python
from orient_express.predictors import UrlImageLoader

loader = UrlImageLoader(rows, url=lambda r: r["image_url"], batch_size=32)
for rows_batch, preds in predictor.predict_stream(loader, confidence=0.4):
    for row, pred in zip(rows_batch, preds):
        ...
```

`decode="fast"` additionally decodes JPEGs at reduced scale sized to the
model's input resolution — measurably faster, but pixels differ from a
full decode: validate model accuracy before enabling it.

**`ImageLoader`** — you provide any per-item `load` callable returning a
PIL image (file read, video frame, crop, custom auth), run on `workers`
threads with bounded look-ahead and per-item fault tolerance. Rule of
thumb for `workers` against a high-latency source: ≈ target img/s ×
per-request seconds (the default 32 covers ~60-70 img/s at typical GCS
latency). If you fetch encoded bytes yourself, `decode_image(data)` is
the fast cv2-backed decode (pixel-identical to PIL for baseline JPEGs):

```python
from orient_express.predictors import ImageLoader, decode_image

loader = ImageLoader(rows, load=lambda r: decode_image(my_bytes(r)),
                     batch_size=32)
```

Either loader fuses with `predict_stream`: each image is resized by the
worker that loaded it, so full-size images never pile up in memory, and
failed items are skipped and reported to `on_error` in both.

Measured on real photos over GCS (dg-otc models, 8-vCPU GCE worker):
threaded `ImageLoader` streams 5-7x over the serial download-then-predict
loop; `UrlImageLoader` sustains ~3-4x `ImageLoader`'s ingest rate on top
of that (see `experiments/streaming_benchmark_*.py` and the download
investigation in the GPU test log).

### Chaining Multiple Models

`map_stream` / `flat_map_stream` are ordered, bounded, threaded stage glue
for multi-model pipelines. Every stage — including predictors — is an
iterable transform, so a detection → crop → embed → search → annotate chain
reads top to bottom and every stage overlaps (measured 5x over the serial
per-photo loop):

```python
from orient_express.predictors import ImageLoader, flat_map_stream, map_stream

# keep_original=True: the payload carries (row, image) pairs so later
# stages can crop from the full-resolution image
loader = ImageLoader(rows, load=download, batch_size=4, keep_original=True)
dets = detector.predict_stream(loader, confidence=0.4)

def crop_stage(batch):                       # one image -> one crop batch
    pairs, det_lists = batch
    for (row, image), d in zip(pairs, det_lists):
        yield (row, image, d), make_crops(image, d)

crops  = flat_map_stream(crop_stage, dets, workers=2)
feats  = extractor.predict_stream(crops)     # second model, batched crops
scored = map_stream(match_pog, feats, workers=4)          # CPU matching
done   = map_stream(annotate_and_upload, scored, workers=8)  # render + IO
for result in done:
    ...
```

### Pinning Model Versions

By default, `get_vertex_model` returns the most recently updated version. To pin to a specific version:

```python
vertex_model = get_vertex_model(
    model_name="my-classifier",
    project_name="my-project",
    region="us-central1",
    version=3,  # Pin to version 3
)
```

---

## Built-in Predictor Types

Orient Express provides four built-in predictor classes for ONNX image models. Each has specific requirements for the ONNX graph structure.

### General ONNX Requirements

All ONNX image models share these requirements:

- **Input images are resized using simple stretch** (no letterboxing/padding) to the model's expected resolution before inference.
- **Normalization must be baked into the ONNX graph.** The library passes uint8 RGB images directly to the model; any normalization (e.g., ImageNet mean/std) must be handled inside the graph.
- **Batch dimension**: Models receive batched inputs with shape `[batch, height, width, 3]`.
- **Score outputs must be probabilities, not logits.** Confidence thresholds are applied to the raw output values, so apply softmax/sigmoid inside the graph.

#### Class ID conventions

The `classes` dict (`{int: str}`) maps model outputs to class names, but the mapping convention differs by predictor type — an off-by-one here produces plausible-looking but wrong labels, so double-check when exporting:

| Predictor type | How `classes` keys are interpreted |
| --- | --- |
| Classification, multi-label | **1-indexed** relative to score columns: class id `N` reads score column `N - 1`. `{1: "cat", 2: "dog"}` means column 0 is cat. |
| Object detection, instance segmentation | The label values the model emits are looked up **directly** as `classes` keys, no offset. |
| Semantic segmentation | The **channel index** of the masks output is looked up directly as a `classes` key (channel 0 ↔ key `0`). |

### ClassificationPredictor

<details>
<summary>Click to expand</summary>

For image classification models that output class probabilities.

#### ONNX Graph Requirements

|             |                                                              |
| ----------- | ------------------------------------------------------------ |
| **Inputs**  | `images`: `[batch, height, width, 3]` uint8 RGB              |
| **Outputs** | `scores`: `[batch, num_classes]` float32 class probabilities |

The graph must handle normalization internally. No target_sizes input is needed.

#### Usage

```python
from orient_express.predictors import ClassificationPredictor

predictor = ClassificationPredictor(
    onnx_path="classifier.onnx",
    classes={1: "cat", 2: "dog", 3: "bird"}
)

predictions = predictor.predict(images)
# Returns: list[ClassificationPrediction]
```

#### Output Structure

```python
@dataclass
class ClassificationPrediction:
    clss: str                      # Predicted class name
    score: float                   # Confidence score for predicted class
    class_scores: dict[str, float] # Scores for all classes

# to_dict() output:
{
    "class": "cat",
    "score": 0.95,
    "class_scores": {"cat": 0.95, "dog": 0.03, "bird": 0.02}
}
```

</details>

### MultiLabelClassificationPredictor

<details>
<summary>Click to expand</summary>

For image multi-label classification models that output a set of binary class probabilities.

#### ONNX Graph Requirements

|             |                                                              |
| ----------- | ------------------------------------------------------------ |
| **Inputs**  | `images`: `[batch, height, width, 3]` uint8 RGB              |
| **Outputs** | `scores`: `[batch, num_classes]` float32 class probabilities |

The graph must handle normalization internally. No target_sizes input is needed.

#### Usage

```python
from orient_express.predictors import MultiLabelClassificationPredictor

predictor = MultiLabelClassificationPredictor(
    onnx_path="classifier.onnx",
    classes={1: "contains_cat", 2: "contains_dog", 3: "contains_bird"}
)

predictions = predictor.predict(images, confidence=0.5)
# Returns: list[MultiLabelClassificationPrediction]
```

#### Output Structure

```python
@dataclass
class MultiLabelClassificationPrediction:
    classes: list[str]             # Predicted class names based on confidence threshold
    class_scores: dict[str, float] # Scores for all classes

# to_dict() output:
{
    "classes": ["contains_cat", "contains_bird"],
    "class_scores": {"contains_cat": 0.95, "contains_dog": 0.03, "contains_bird": 0.82}
}
```

</details>

### BoundingBoxPredictor

<details>
<summary>Click to expand</summary>

For object detection models that output bounding boxes.

#### ONNX Graph Requirements

|             |                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------- |
| **Inputs**  | `images`: `[batch, height, width, 3]` uint8 RGB                                                   |
|             | `target_sizes`: `[batch, 2]` float32 containing `[height, width]` of original images              |
| **Outputs** | `boxes`: `[batch, num_detections, 4]` float32 as `[x1, y1, x2, y2]` in original image coordinates |
|             | `scores`: `[batch, num_detections]` float32 confidence scores                                     |
|             | `labels`: `[batch, num_detections]` int64 class indices                                           |

The ONNX graph must rescale bounding boxes to the original image dimensions using `target_sizes`. The library does not perform any box coordinate transformation.

#### Usage

```python
from orient_express.predictors import BoundingBoxPredictor

predictor = BoundingBoxPredictor(
    onnx_path="detector.onnx",
    classes={1: "person", 2: "car", 3: "bicycle"}
)

predictions = predictor.predict(images, confidence=0.5, nms_threshold=0.4)
# Returns: list[list[BoundingBoxPrediction]]
# Outer list: per image, inner list: detections for that image
```

#### Output Structure

```python
@dataclass
class BoundingBoxPrediction:
    clss: str           # Class name
    score: float        # Confidence score
    bbox: np.ndarray    # [x1, y1, x2, y2] in original image coordinates

# to_dict() output:
{
    "class": "person",
    "score": 0.92,
    "bbox": {"x1": 100.5, "y1": 50.2, "x2": 300.8, "y2": 400.1}
}
```

#### Annotation

```python
annotated_image = predictor.get_annotated_image(image, predictions[0])
# Returns PIL.Image with bounding boxes drawn
```

</details>

### InstanceSegmentationPredictor

<details>
<summary>Click to expand</summary>

For instance segmentation models that output bounding boxes and per-instance masks.

#### ONNX Graph Requirements

|             |                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------- |
| **Inputs**  | `images`: `[batch, height, width, 3]` uint8 RGB                                                   |
|             | `target_sizes`: `[batch, 2]` float32 containing `[height, width]` of original images              |
| **Outputs** | `boxes`: `[batch, num_detections, 4]` float32 as `[x1, y1, x2, y2]` in original image coordinates |
|             | `scores`: `[batch, num_detections]` float32 confidence scores                                     |
|             | `labels`: `[batch, num_detections]` int64 class indices                                           |
|             | `masks`: `[batch, num_detections, mask_height, mask_width]` float32 mask logits                   |

The ONNX graph must rescale bounding boxes to original image dimensions using `target_sizes`. Masks can be any resolution—they are resized to original image dimensions in Python postprocessing using bilinear interpolation.

#### Usage

```python
from orient_express.predictors import InstanceSegmentationPredictor

predictor = InstanceSegmentationPredictor(
    onnx_path="instance_seg.onnx",
    classes={1: "person", 2: "car", 3: "bicycle"}
)

predictions = predictor.predict(images, confidence=0.5)
# Returns: list[list[InstanceSegmentationPrediction]]
```

#### Output Structure

```python
@dataclass
class InstanceSegmentationPrediction:
    clss: str           # Class name
    score: float        # Confidence score
    bbox: np.ndarray    # [x1, y1, x2, y2] in original image coordinates
    mask: np.ndarray    # Boolean mask at original image resolution

# to_dict(include_mask=False) output:
{
    "class": "person",
    "score": 0.89,
    "bbox": {"x1": 100.5, "y1": 50.2, "x2": 300.8, "y2": 400.1}
}

# to_dict(include_mask=True) adds:
{
    ...
    "mask": [[True, True, False, ...], ...]  # 2D boolean list
}
```

#### Annotation

```python
annotated_image = predictor.get_annotated_image(image, predictions[0])
# Returns PIL.Image with mask overlays and instance indices
```

</details>

### SemanticSegmentationPredictor

<details>
<summary>Click to expand</summary>

For semantic segmentation models that output per-pixel class predictions.

#### ONNX Graph Requirements

|             |                                                                               |
| ----------- | ----------------------------------------------------------------------------- |
| **Inputs**  | `images`: `[batch, height, width, 3]` uint8 RGB                               |
| **Outputs** | `masks`: `[batch, num_classes, mask_height, mask_width]` float32 class probabilities |

Masks can be any resolution—they are resized to original image dimensions in Python postprocessing. The class dimension is reduced via argmax to produce a single class ID per pixel. The output values must be probabilities (softmax/sigmoid inside the graph), because the per-pixel validity mask thresholds the max class probability against the `confidence` parameter.

#### Usage

```python
from orient_express.predictors import SemanticSegmentationPredictor

predictor = SemanticSegmentationPredictor(
    onnx_path="semantic_seg.onnx",
    classes={0: "background", 1: "road", 2: "building", 3: "vegetation"}
)

predictions = predictor.predict(images)
# Returns: list[SemanticSegmentationPrediction]
```

#### Output Structure

```python
@dataclass
class SemanticSegmentationPrediction:
    class_mask: np.ndarray   # [height, width] int array of class indices
    conf_masks: np.ndarray   # [num_classes, height, width] float confidence per class

# to_dict(include_conf_masks=False) output:
{
    "class_mask": [[0, 0, 1, 2, ...], ...]  # 2D int array
}

# to_dict(include_conf_masks=True) adds:
{
    ...
    "conf_masks": [[[0.1, 0.2, ...], ...], ...]  # 3D float array
}
```

#### Annotation

```python
annotated_image = predictor.get_annotated_image(image, predictions[0].class_mask)
# Returns PIL.Image with color-coded segmentation overlay
```

</details>

### VectorIndex

<details>
<summary>Click to expand</summary>

A cosine-similarity vector index for matching feature vectors to labels. Each vector in the index can have one or more labels. VectorIndex integrates with `get_predictor` for loading from saved artifacts, and can be built from scratch using a feature extractor.

#### Usage

```python
from orient_express.predictors import VectorIndex, build_vector_index

# Build from crops and labels using a feature extractor
index = build_vector_index(
    crops=crop_images,           # list of PIL Images or file paths
    labels=cluster_ids,          # one label per crop
    feature_extractor=fe,        # FeatureExtractionPredictor
    num_workers=8,               # parallel image loading
)

# Save and load
index.dump("/path/to/artifact_dir")

from orient_express.predictors import get_predictor
loaded_index = get_predictor("/path/to/artifact_dir")

# Search
results = loaded_index.search(query_vector, k=5)
for result in results:
    print(result.labels, result.score)

# Batch search
batch_results = loaded_index.search_batch(query_matrix, k=5)
```

#### Multi-label support

Vectors can have composite labels (use tuples). This is useful when a single visual cluster maps to multiple things:

```python
index = VectorIndex(
    vectors=feature_matrix,
    labels=[("sku_101", "sku_102"), ("sku_103",)],
)
```

#### Per-label aggregation

Vector indices in which labels are not unique can be aggregated so that each label has a single centroid.
If `per_label=True` and the labels are composite (tuples), then the labels will be unpacked and aggregated separately.

```python
aggregated = index.aggregate(per_label=True)  # 3 vectors, one per label element ["sku_101", "sku_102", "sku_103"]
aggregated = index.aggregate(per_label=False)  # 2 vectors, one per composite label  [("sku_101", "sku_102"), ("sku_103")]
```

#### Output Structure

```python
@dataclass
class SearchResult:
    labels: list   # All labels for the matched vector
    score: float   # Cosine similarity score
```

</details>

---

## Deployed Endpoint APIs

When you upload a model with orient-express and deploy it to a Vertex AI endpoint, the actual HTTP API exposed by the endpoint is determined by the serving container image — not by your Python predictor code. Orient-express ships two such images:

- `image-onnx` — serves the built-in ONNX predictor types (classification, detection, segmentation).
- `xgboost-scikit-learn` — serves any joblib-loadable model (sklearn pipelines, xgboost, etc.).

This section documents the request/response shape each image's endpoint exposes once deployed.

### How the docker images connect to GCP endpoints

The deployment flow is:

1. **Train + export.** You build a predictor locally (e.g. `ClassificationPredictor("model.onnx", classes)`) or train a sklearn/xgboost model.
2. **Upload.** `upload_model` / `upload_model_joblib` pushes the artifacts to GCS under `gs://<bucket>/models/<model_name>/<version>/` and registers a Vertex AI Model with `serving_container_image_uri` pointing at one of orient-express's images.
3. **Deploy.** `vertex_model.deploy_to_endpoint(...)` (or the Vertex console) attaches the registered model to a Vertex AI Endpoint. Vertex starts the container with `AIP_STORAGE_URI` set to the GCS path from step 2, plus `MODEL_NAME` set to the model's display name.
4. **Serve.** The container downloads the artifacts on startup, instantiates the right predictor via metadata, and listens on `/v1/models/<MODEL_NAME>:predict`.
5. **Call.** Clients POST to `https://<region>-aiplatform.googleapis.com/v1/projects/<project>/locations/<region>/endpoints/<endpoint_id>:predict` with a Bearer token. Vertex routes the request into the container and returns the JSON response.

Every endpoint accepts the same envelope:

```json
{
  "instances": [...],
  "parameters": {...}
}
```

What goes in `instances` / `parameters`, and what comes back in `predictions`, is per-image and per-predictor — covered below.

### ONNX Image Endpoint

Image: `us-west1-docker.pkg.dev/shiftsmart-api/orient-express/image-onnx:<tag>`

Common request shape across all ONNX predictor types:

```json
{
  "instances": [
    {"image": "<http(s) URL | gs:// URI | base64 | data URI>"}
  ],
  "parameters": {
    "confidence": 0.5
  }
}
```

Common response envelope:

```json
{
  "predictions": [
    {"status": "success", ...predictor-specific fields...}
  ]
}
```

`status` values: `"success"`, `"failed to download image"`, or `"failed to get debug image"`. Malformed payloads return top-level `{"error": "Failed to decode input"}` instead of `predictions`.

The predictor-specific fields differ by model type — covered in the subsections below.

#### Classification

For models uploaded as `ClassificationPredictor`. `parameters.confidence` is **not** honored (the predictor always returns the top class).

Per-image response:

```json
{
  "status": "success",
  "class": "cat",
  "score": 0.95,
  "class_scores": {"cat": 0.95, "dog": 0.03, "bird": 0.02}
}
```

No `debug_image` for classification (nothing meaningful to draw).

#### Multi-label classification

For models uploaded as `MultiLabelClassificationPredictor`. `parameters.confidence` is the per-class threshold for inclusion in `classes` (default `0.5`).

Per-image response:

```json
{
  "status": "success",
  "predictions": {
    "classes": ["contains_cat", "contains_bird"],
    "class_scores": {"contains_cat": 0.95, "contains_dog": 0.03, "contains_bird": 0.82}
  },
  "debug_image": null
}
```

`debug_image` is always `null` for multi-label.

#### Object detection

For models uploaded as `BoundingBoxPredictor`. `parameters.confidence` filters detections below the threshold (default `0.5`).

Per-image response:

```json
{
  "status": "success",
  "predictions": [
    {"class": "person", "score": 0.92, "bbox": {"x1": 100.5, "y1": 50.2, "x2": 300.8, "y2": 400.1}}
  ],
  "debug_image": "<base64 JPEG with boxes overlaid>"
}
```

`bbox` coordinates are in pixels of the original (EXIF-corrected) image. `predictions` is an empty list when nothing clears the confidence threshold.

#### Instance segmentation

For models uploaded as `InstanceSegmentationPredictor`. `parameters.confidence` filters detections (default `0.5`).

Per-image response:

```json
{
  "status": "success",
  "predictions": [
    {"class": "person", "score": 0.89, "bbox": {"x1": 100.5, "y1": 50.2, "x2": 300.8, "y2": 400.1}}
  ],
  "debug_image": "<base64 JPEG with masks overlaid>"
}
```

Per-instance mask arrays are **not** included in the response by default (too large). The annotated mask overlay is baked into `debug_image`.

#### Semantic segmentation

For models uploaded as `SemanticSegmentationPredictor`. `parameters.confidence` is the per-pixel threshold above which a class is considered "valid" (default `0.5`).

Per-image response:

```json
{
  "status": "success",
  "predictions": {
    "class_mask": "<base64 PNG, uint8, per-pixel class id>",
    "valid_mask": "<base64 PNG, uint8, 0=below threshold, 1=above>"
  },
  "debug_image": "<base64 JPEG with color-coded overlay>"
}
```

`class_mask` always paints every pixel with the argmax winner. `valid_mask` tells you which pixels actually cleared the confidence threshold — AND them together client-side to get the "real" segmentation.

### XGBoost / scikit-learn Endpoint

Image: `us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn:<tag>`

For models uploaded via `upload_model_joblib` — sklearn pipelines, xgboost models, or anything `joblib.load`-able with a `.predict(DataFrame)` method.

Request shape — `instances` is a list of dicts, one row per input:

```json
{
  "instances": [
    {"pclass": 1, "sex": "female", "age": 29, "fare": 100.0, "embarked": "S"},
    {"pclass": 3, "sex": "male", "age": 35, "fare": 8.05, "embarked": "S"}
  ]
}
```

The server constructs `pd.DataFrame(instances)` and calls `model.predict(df)` on it. The columns your pipeline expects must be present in each instance dict.

Response shape — one prediction per input row:

```json
{
  "predictions": [0, 1]
}
```

Each element is whatever your `.predict()` returns — a class label for classifiers, a numeric value for regressors, an array for multi-output models.

`parameters` is ignored — there's no per-request configuration for this image.

---

## Color Schemes

For predictors that support annotation (`BoundingBoxPredictor`, `InstanceSegmentationPredictor`, `SemanticSegmentationPredictor`), you can set a custom color scheme:

```python
predictor.color_scheme = {
    "person": (255, 0, 0),    # Red (RGB)
    "car": (0, 255, 0),       # Green
    "bicycle": (0, 0, 255),   # Blue
}
```

Colors are specified as RGB tuples.

## Legacy API (removed in 3.0)

The `ModelExpress` / `JoblibSimpleLoader` wrapper API and the
`orient_express.deployment` / `orient_express.sklearn_pipeline` modules were
removed in v3.0.0. Everything up to and including the last 2.4.x release keeps
the legacy API — install from PyPI (`orient_express<3`) or from a pre-3.0 git
ref if you still need it. The joblib workflow above (`upload_model_joblib` +
`get_vertex_model`) is the replacement.
