# Orient Express
A library to accelerate model deployments to Vertex AI directly from colab notebooks

![train-resized](https://github.com/user-attachments/assets/f1ed32ec-07d9-4d48-8b96-3323db6b5091)

Orient Express is a library designed to streamline the deployment of ONNX-based Computer Vision models to Vertex AI. It standardizes the wrapping of models into Predictors, handles interaction with the Vertex Model Registry, and simplifies both local and remote inference workflows.

## Installation

```bash
pip install orient_express
```

## Supported Model Types

The library currently supports the following computer vision tasks. Each corresponds to a specific `Predictor` class in the library:

* **Classification** (`ClassificationPredictor`)
* **Object Detection** (`BoundingBoxPredictor`)
* **Instance Segmentation** (`InstanceSegmentationPredictor`)
* **Semantic Segmentation** (`SemanticSegmentationPredictor`)

## ONNX Graph Requirements

To ensure compatibility with `orient_express`, your exported ONNX graphs must adhere to strict input/output signatures and internal processing rules.

### General Preprocessing Rules
* **Normalization**: The library **does not** perform normalization (e.g., mean subtraction, std division) in Python. Your ONNX graph must accept raw pixel values (0-255) and handle normalization internally (e.g., Cast to Float -> Divide by 255 -> Normalize).
* **Resizing**: The library resizes input images to the model's expected `resolution` using a **stretch** resize (not letterbox/pad). The input tensor provided to the graph will be `(Batch, Resolution, Resolution, 3)`. If your model requires `NCHW`, the graph must handle the transpose.

### 1. Classification
* **Inputs**:
    * `images`: `(B, H, W, 3)` | **Dtype**: `uint8`
* **Outputs**:
    * `scores`: Class scores/logits.

### 2. Object Detection
* **Inputs**:
    * `images`: `(B, H, W, 3)` | **Dtype**: `uint8`
    * `target_sizes`: `(B, 2)` containing original image `(height, width)` | **Dtype**: `float32`
* **Outputs**:
    * `boxes`: `(B, N, 4)` coordinates `[x1, y1, x2, y2]`. **Crucial**: The graph must rescale these boxes to the original dimensions provided in `target_sizes`.
    * `scores`: `(B, N)`
    * `labels`: `(B, N)` Class indices.

### 3. Instance Segmentation
* **Inputs**:
    * `images`: `(B, H, W, 3)` | **Dtype**: `uint8`
    * `target_sizes`: `(B, 2)` containing original image `(height, width)` | **Dtype**: `float32`
* **Outputs**:
    * `boxes`: `(B, N, 4)`. **Crucial**: Must be rescaled to original dimensions inside the graph.
    * `scores`: `(B, N)`
    * `labels`: `(B, N)`
    * `masks`: `(B, N, H_mask, W_mask)`. Raw mask outputs. The library handles resizing these masks to the original image size during post-processing.

### 4. Semantic Segmentation
* **Inputs**:
    * `images`: `(B, H, W, 3)` | **Dtype**: `uint8`
    * *Note: Does not accept `target_sizes`.*
* **Outputs**:
    * `masks`: `(1, Num_Classes, H_mask, W_mask)` or similar. The library handles resizing the output masks to match the input image size in post-processing.

---

## Workflow 1: Export & Upload Model

This is the primary entry point. You must instantiate a local predictor with your ONNX model and upload it to the Vertex AI Model Registry.

```python
from orient_express.predictors import InstanceSegmentationPredictor
from orient_express.vertex import upload_model

# 1. Define your class mapping (ID -> Name)
classes = {
    1: "person",
    2: "bicycle",
    3: "car"
}

# 2. Instantiate the local predictor
# This wraps your ONNX file and handles standardized post-processing
local_predictor = InstanceSegmentationPredictor(
    model_path="path/to/my_model.onnx", 
    classes=classes
)

# 3. Upload to Vertex AI Model Registry
# This dumps the necessary metadata and artifacts and registers the model
vertex_model = upload_model(
    model=local_predictor,
    model_name="traffic-segmentation",
    project_name="my-gcp-project",
    region="us-central1",
    bucket_name="my-artifact-bucket"
)

print(f"Model uploaded: {vertex_model.name} version {vertex_model.version}")
```

## Workflow 2: Local Inference

> **Prerequisite:** This workflow **only** works for models that have already been uploaded to the registry via [Workflow 1](#workflow-1-export--upload-model).

You can pull a model from the registry to run inference on your local machine without deploying a remote endpoint.

```python
from orient_express.vertex import get_vertex_model
from PIL import Image

# 1. Fetch the model reference from Vertex Registry
vertex_model = get_vertex_model(
    model_name="traffic-segmentation",
    project_name="my-gcp-project",
    region="us-central1"
)

# 2. Download artifacts and instantiate the predictor locally
predictor = vertex_model.get_local_predictor()

# 3. Run Inference
image = Image.open("street.jpg")
predictions = predictor.predict(
    images=[image], 
    confidence=0.5
)

# 4. Visualization (Optional)
debug_image = predictor.get_annotated_image(image, predictions[0])
debug_image.show()
```

## Workflow 3: Remote Inference (Online Endpoint)

> **Prerequisite:** This workflow **only** works for models that have already been uploaded to the registry via [Workflow 1](#workflow-1-export--upload-model).

Deploy the model to a Vertex AI Endpoint and run inference via API calls.

```python
from orient_express.vertex import get_vertex_model

# 1. Get the model
vertex_model = get_vertex_model(
    model_name="traffic-segmentation",
    project_name="my-gcp-project",
    region="us-central1"
)

# 2. Deploy to an Endpoint
# This creates a Vertex Endpoint and deploys the model container
vertex_model.deploy_to_endpoint(
    endpoint_name="traffic-seg-endpoint",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=2
)

# 3. Remote Prediction
# Orient Express handles image encoding/decoding automatically
instances = [
    {"image": "gs://my-bucket/street.jpg"}, # GCS URI
    {"image": "http://example.com/car.jpg"} # HTTP URL
]

results = vertex_model.remote_predict(
    instances=instances,
    endpoint_name="traffic-seg-endpoint"
)

print(results)
```

## Prediction Return Types

The `predict()` methods return standardized dataclasses depending on the predictor type.

### `ClassificationPrediction`
* `clss` (str): The name of the highest scoring class.
* `score` (float): The confidence score of the highest class.
* `class_scores` (dict[str, float]): Dictionary of all classes and their scores.

### `BoundingBoxPrediction`
* `clss` (str): Detected class name.
* `score` (float): Confidence score.
* `bbox` (np.ndarray): Array `[x1, y1, x2, y2]` (absolute coordinates).

### `InstanceSegmentationPrediction`
* `clss` (str): Detected class name.
* `score` (float): Confidence score.
* `bbox` (np.ndarray): Array `[x1, y1, x2, y2]`.
* `mask` (np.ndarray): Boolean array representing the segmentation mask (same size as original image).

### `SemanticSegmentationPrediction`
* `class_mask` (np.ndarray): 2D array where each pixel value corresponds to a class ID.
* `conf_masks` (np.ndarray): Raw confidence masks.
## Installation

```
pip install orient_express
```

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
