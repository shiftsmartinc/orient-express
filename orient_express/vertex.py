import os
import tempfile

import yaml
import joblib
from google.cloud import storage, aiplatform

from .predictors import get_predictor, Predictor


ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_vertex_initialized = False


class VertexModel:
    def __init__(
        self,
        vertex_model,
        model_name: str,
        project_name: str,
        region: str,
        version: int,
    ):
        self.vertex_model = vertex_model
        self.name = vertex_model.name
        self.model_name = model_name
        self.project_name = project_name
        self.region = region
        self.version = version

    def deploy_to_endpoint(
        self,
        endpoint_name: str,
        machine_type: str,
        min_replica_count: int,
        max_replica_count: int,
    ):
        endpoint = self.get_or_create_endpoint(endpoint_name)
        self.vertex_model.deploy(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            traffic_percentage=100,
        )

    def get_or_create_endpoint(self, endpoint_name: str):
        endpoint = self.get_endpoint(endpoint_name)
        if endpoint:
            return endpoint
        else:
            return self.create_endpoint(endpoint_name)

    def get_endpoint(self, endpoint_name: str):
        endpoints = aiplatform.Endpoint.list(
            filter=f"display_name={endpoint_name}", order_by="create_time"
        )
        if endpoints:
            return endpoints[0]

    def create_endpoint(self, endpoint_name: str):
        return aiplatform.Endpoint.create(
            display_name=endpoint_name,
            project=self.project_name,
            location=self.region,
        )

    def remote_predict(self, instances: list, endpoint_name: str):
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            raise Exception(
                f"Endpoint '{endpoint_name}' not found. Please deploy the model first."
            )
        self.endpoint = endpoint

        predictions = self.endpoint.predict(instances=instances)
        return predictions.predictions

    def get_local_predictor(self):
        dir = os.path.join(ARTIFACT_DIR, self.model_name + "-" + str(self.version))
        self.download_artifacts(dir)
        return get_predictor(dir)

    def download_artifacts(self, dir: str):
        download_artifacts(dir, self.vertex_model.gca_resource.artifact_uri)


def vertex_init(project_name: str, region: str):
    global _vertex_initialized
    if not _vertex_initialized:
        aiplatform.init(project=project_name, location=region)
        _vertex_initialized = True


def download_artifacts(dir: str, artifact_uri: str):
    storage_client = storage.Client()

    bucket_name, artifact_path = artifact_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)

    os.makedirs(dir, exist_ok=True)

    blobs = bucket.list_blobs(prefix=artifact_path)
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        download_path = os.path.join(dir, filename)
        blob.download_to_filename(download_path)


def upload_model(
    model: Predictor,
    model_name: str,
    project_name: str,
    region: str,
    bucket_name: str,
    serving_container_image_uri: str = "",
    serving_container_health_route: str = "",
    serving_container_predict_route: str = "",
    labels: dict[str, str] | None = {},
):
    """
    Upload a Predictor model to Vertex AI Model Registry.

    Args:
        model: Any joblib-serializable model
        model_name: Display name for the model in the registry
        project_name: GCP project ID
        region: GCP region (e.g., 'us-central1')
        bucket_name: GCS bucket for storing model artifacts
        serving_container_image_uri: Docker image URI for serving the model
        serving_container_health_route: Health check endpoint route
        serving_container_predict_route: Prediction endpoint route
        labels: Optional labels to attach to the model

    Returns:
        VertexModel instance
    """
    if not serving_container_image_uri:
        serving_container_image_uri = model.get_serving_container_image_uri()
    if not serving_container_health_route:
        serving_container_health_route = model.get_serving_container_health_route(
            model_name
        )
    if not serving_container_predict_route:
        serving_container_predict_route = model.get_serving_container_predict_route(
            model_name
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        file_list = model.dump(temp_dir)
        vertex_model = upload_model_with_files(
            file_list,
            model_name,
            project_name,
            region,
            bucket_name,
            serving_container_image_uri,
            serving_container_health_route,
            serving_container_predict_route,
            labels,
        )
    return vertex_model


def upload_model_joblib(
    model,
    model_name: str,
    project_name: str,
    region: str,
    bucket_name: str,
    serving_container_image_uri: str,
    serving_container_health_route: str,
    serving_container_predict_route: str,
    labels: dict[str, str] | None = {},
):
    """
    Upload a joblib-serializable model to Vertex AI Model Registry.

    Unlike upload_model which works with Predictor instances, this function
    accepts any model that can be serialized with joblib (e.g., scikit-learn
    pipelines, XGBoost models).

    Args:
        model: Any joblib-serializable model
        model_name: Display name for the model in the registry
        project_name: GCP project ID
        region: GCP region (e.g., 'us-central1')
        bucket_name: GCS bucket for storing model artifacts
        serving_container_image_uri: Docker image URI for serving the model
        serving_container_health_route: Health check endpoint route
        serving_container_predict_route: Prediction endpoint route
        labels: Optional labels to attach to the model

    Returns:
        VertexModel instance
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.joblib")
        metadata_path = os.path.join(temp_dir, "metadata.yaml")

        joblib.dump(model, model_path)

        metadata = {
            "model_type": "joblib",
            "model_file": "model.joblib",
        }
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)

        file_list = [metadata_path, model_path]

        vertex_model = upload_model_with_files(
            file_list,
            model_name,
            project_name,
            region,
            bucket_name,
            serving_container_image_uri,
            serving_container_health_route,
            serving_container_predict_route,
            labels,
        )
    return vertex_model


def upload_model_with_files(
    file_list: list[str],
    model_name: str,
    project_name: str,
    region: str,
    bucket_name: str,
    serving_container_image_uri: str,
    serving_container_health_route: str,
    serving_container_predict_route: str,
    labels: dict[str, str] | None = {},
) -> VertexModel:
    parent_model = get_vertex_model(
        model_name, project_name, region, raise_exception=False
    )
    if parent_model:
        version = parent_model.version + 1
    else:
        version = 1

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    artifact_dir = f"models/{model_name}/{version}/"
    for file_name in file_list:
        blob = bucket.blob(f"{artifact_dir}{file_name}")
        blob.upload_from_filename(file_name)

    artifact_uri = f"gs://{bucket_name}/{artifact_dir}"

    if parent_model:
        parent_model_uri = (
            f"projects/{project_name}/locations/{region}/models/{parent_model.name}"
        )
    else:
        parent_model_uri = None

    release = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=artifact_uri,
        parent_model=parent_model_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_health_route=serving_container_health_route,
        serving_container_predict_route=serving_container_predict_route,
        sync=True,
        labels=labels,
        serving_container_environment_variables={"MODEL_NAME": model_name},
    )

    return VertexModel(release, model_name, project_name, region, version)


def get_vertex_model(
    model_name: str,
    project_name: str,
    region: str,
    version: int | None = None,
    raise_exception: bool = True,
):
    vertex_init(project_name, region)
    models = aiplatform.Model.list(filter=f"display_name={model_name}")

    if not models:
        if raise_exception:
            raise Exception(
                f"Model '{model_name}' not found in registry for project '{project_name}' region '{region}'"
            )
        else:
            return None

    if version is None:
        latest_model = sorted(models, key=lambda x: x.update_time, reverse=True)[0]
        return VertexModel(
            latest_model, model_name, project_name, region, int(latest_model.version_id)
        )

    for model in models:
        if int(model.version_id) == version:
            return VertexModel(model, model_name, project_name, region, version)

    if raise_exception:
        raise Exception(
            f"Model '{model_name}' with version '{version}' not found in registry for project '{project_name}' region '{region}'"
        )
    return None
