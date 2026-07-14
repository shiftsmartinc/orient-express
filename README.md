# Orient Express

A library to accelerate model deployments to Vertex AI directly from colab notebooks

![train-resized](https://github.com/user-attachments/assets/f1ed32ec-07d9-4d48-8b96-3323db6b5091)

Orient Express provides two main capabilities:

1. **Vertex Model Deployment and Retrieval**: Capabilities for uploading, downloading, or deploying models to Vertex AI Model Registry.

1. **ONNX Image Model Deployment**: Built-in predictor classes for easily running image classification, object detection, instance segmentation, and semantic segmentation models exported to ONNX format.

Both workflows handle versioning, artifact storage in GCS, and integration with Vertex AI Model Registry.

## Installation

```bash
pip install orient_express
```

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

### Platform Support Matrix

| Platform | Architecture | ONNX Runtime Package | CUDA Available |
| -------- | ------------ | -------------------- | -------------- |
| Linux    | x86_64       | onnxruntime-gpu      | Yes            |
| Linux    | aarch64      | onnxruntime          | No             |
| Windows  | x64 (AMD64)  | onnxruntime-gpu      | Yes            |
| Windows  | ARM64        | onnxruntime          | No             |
| macOS    | x86_64       | onnxruntime          | No             |
| macOS    | arm64        | onnxruntime          | No             |

The appropriate package is installed automatically based on your platform.

### Selecting CPU vs CUDA Execution

When loading a predictor, use the `device` parameter to specify the execution provider:

```python
from orient_express.predictors import ObjectDetectionPredictor

# CPU inference (works on all platforms)
predictor = ObjectDetectionPredictor("/path/to/model", classes, device="cpu")

# CUDA inference (requires Linux x64 or Windows x64 with CUDA drivers)
predictor = ObjectDetectionPredictor("/path/to/model", classes, device="cuda")
```

When using a Vertex AI model:

```python
# CPU inference
predictor = model.get_local_predictor(device="cpu")

# CUDA inference
predictor = model.get_local_predictor(device="cuda")
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
