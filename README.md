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

For local development:

```bash
pip install -e .
```

Or with Poetry:

```bash
poetry install
```

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
    [{"image": "https://storage.googleapis.com/ssm-media-uploads/example.jpg"}],
    endpoint_name="my-classifier-endpoint"
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
| **Outputs** | `masks`: `[batch, num_classes, mask_height, mask_width]` float32 class logits |

Masks can be any resolution—they are resized to original image dimensions in Python postprocessing. The class dimension is reduced via argmax to produce a single class ID per pixel.

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
    labels=[("sku_101", "sku_102"), ("sku_103")],
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

## Legacy API [Still Maintained]

<details>
<summary>Click to expand</summary>

## Example

### Train Model

Train a regular model. In the example below, it's xgboost model, trained on the Titanic dataset.

```python

# Import necessary libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the Titanic dataset
data = sns.load_dataset('titanic').dropna(subset=['survived'])  # Dropping rows with missing target labels

# Select features and target
X = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = data['survived']

# Define preprocessing for numeric columns (impute missing values and scale features)
numeric_features = ['age', 'fare', 'sibsp', 'parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical columns (impute missing values and one-hot encode)
categorical_features = ['pclass', 'sex', 'embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that first transforms the data, then trains an XGBoost model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)
```

## Upload Model To Model Registry

```python

model_wrapper  = ModelExpress(model=model,
                             project_name='my-project-name',
                             region='us-central1',
                             bucket_name='my-artifacts-bucket',
                             model_name='titanic')
model_wrapper.upload()
```

## Local Inference (Without Online Prediction Endpoint)

The following code will download the last model from the model registry and run the inference locally.

```python

# create input dataframe
titanic_data = {
    "pclass": [1],          # Passenger class (1st, 2nd, 3rd)
    "sex": ["female"],      # Gender
    "age": [29],            # Age
    "sibsp": [0],           # Number of siblings/spouses aboard
    "parch": [0],           # Number of parents/children aboard
    "fare": [100.0],        # Ticket fare
    "embarked": ["S"]       # Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
}
input_df = pd.DataFrame(titanic_data)

# init the model wrapper
model_wrapper  = ModelExpress(project_name='my-project-name',
                             region='us-central1',
                             model_name='titanic')

# Run inference locally
# It will download the most recent version from the model registry automatically
model_wrapper.local_predict(input_df)
```

## Pin Model Version

In many cases, the pipeline should be pinned to a specific model version so the model can only
be updated explicitly. Just pass a `model_version` parameter when instantiating the ModelExpress wrapper.

```python

# init the model wrapper
model_wrapper  = ModelExpress(project_name='my-project-name',
                             region='us-central1',
                             model_name='titanic',
                             model_version=11)
```

## Remote Inference (With Online Prediction Endpoint)

Make sure the model is deployed:

```python

model_wrapper  = ModelExpress(model=model,
                             project_name='my-project-name',
                             region='us-central1',
                             bucket_name='my-artifacts-bucket',
                             model_name='titanic')

# upload the version to the registry and deploy it to the endpoint
model_wrapper.deploy()
```

Run inference with `remote_predict` method. It will make a remote call to the endpoint without fetching the model locally.

```python

titanic_data = {
    "pclass": [1],             # Passenger class (1st, 2nd, 3rd)
    "sex": ["female"],         # Gender
    "age": [29],               # Age
    "sibsp": [0],              # Number of siblings/spouses aboard
    "parch": [0],              # Number of parents/children aboard
    "fare": [100.0],           # Ticket fare
    "embarked": ["S"]          # Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
}
df = pd.DataFrame(titanic_data)

model_wrapper.remote_predict(df)
```

## Pipeline Deployment Function

Orient express library also have a helper function to simplify Vertex AI pipeline deployment.

Create `deploy.py` script

```python

from orient_express.deployment import deploy_pipeline

import argparse
import conf

from pipeline import pipeline
from orient_express.deployment import deploy_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-type", required=True)

    args = parser.parse_args()
    deploy_pipeline(run_type=args.run_type,
                    pipeline_dsl=pipeline,
                    pipeline_root=conf.PIPELINE_ROOT,
                    pipeline_name=conf.PIPELINE_NAME,
                    pipeline_display_name=conf.PIPELINE_DISPLAY_NAME,
                    pipeline_schedule_name=conf.SCHEDULE_NAME,
                    gcp_project=conf.PROJECT_ID,
                    gcp_location='us-central1',
                    gcp_service_account=conf.SERVICE_ACCOUNT,
                    gcp_network=conf.NETWORK_NAME,
                    gcp_labels={"team": "ml"})
```

And conf.py, make sure to replace the sample values with yours.

```python

import os

BASE_PATH = "gs://pipelines-bucket/vertex-ai/pipelines"

PIPELINE_NAME = "my-pipeline"
PIPELINE_ROOT = f"{BASE_PATH}/{PIPELINE_NAME}"
PIPELINE_TEMP_ROOT = f"{BASE_PATH}/{PIPELINE_NAME}-temp"

PIPELINE_DISPLAY_NAME = "My Pipeline"
PIPELINE_DESCRIPTION = "My example pipeline"

NETWORK_NAME = "project network id"

DOCKER_IMAGE = "us-docker.pkg.dev/my-project/my-artifactory/my-pipeline:latest
BASE_IMAGE = "python:3.11"
PROJECT_ID = "my-project"
PROJECT_REGION = "us-central1"

SERVICE_ACCOUNT = "my-service-account@my-project.iam.gserviceaccount.com"
SCHEDULE_NAME = "My Pipeline"
```

For testing it on a local machine, make sure to authorize to GCP first

```shell

gcloud auth application-default login

```

Finally, run the pipeline (it will execute once)

```shell

python deploy.py --run-type single-run
```

Or, create a scheduler to run continuously

```shell

python deploy.py --run-type scheduled
```

</details>
