import os
from typing import Optional

import joblib
import logging
import pandas as pd
from google.cloud import storage
from google.cloud import aiplatform


class BaseLoader:

    def dump(self) -> list[str]: ...

    def load(self) -> object: ...


class JoblibSimpleLoader(BaseLoader):
    def __init__(self, model=None, serialized_model_path="model.joblib"):
        self.serialized_model_path = serialized_model_path
        self.model = model

    def dump(self) -> list[str]:
        """
        Save model locally, and return a list of local files
        """
        joblib.dump(self.model, self.serialized_model_path)
        return [self.serialized_model_path]

    def load(self):
        return joblib.load(self.serialized_model_path)


class ModelExpress:
    def __init__(
        self,
        model_name: str,
        project_name: str,
        bucket_name: str = None,
        model_version: Optional[int] = None,
        model: object = None,
        region: str = "us-central1",
        model_loader: BaseLoader = None,
        serving_container_image_uri: str = "us-west1-docker.pkg.dev/shiftsmart-api/orient-express/xgboost-scikit-learn:latest",
        serving_container_predict_route: str = "/v1/models/orient-express-model:predict",
        serving_container_health_route: str = "/v1/models/orient-express-model",
        endpoint_name: Optional[str] = None,
        machine_type="n1-standard-4",
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        labels: dict[str, str] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.region = region
        self.project_name = project_name
        self.bucket_name = bucket_name

        if model_loader is None:
            self.model_loader = JoblibSimpleLoader(model=model)

        self.serving_container_image_uri = serving_container_image_uri
        self.serving_container_predict_route = serving_container_predict_route
        self.serving_container_health_route = serving_container_health_route
        self.machine_type = machine_type
        self.min_replica_count = min_replica_count
        self.max_replica_count = max_replica_count
        self.endpoint = None

        if not endpoint_name:
            self.endpoint_name = f"orient-express-{model_name}"
        else:
            self.endpoint_name = endpoint_name

        self._vertex_initialized = False
        self.labels = labels

    def colab_auth(self):
        from google.colab import auth

        auth.authenticate_user()

    def _vertex_init(self):
        if not self._vertex_initialized:
            aiplatform.init(project=self.project_name, location=self.region)
            self._vertex_initialized = True

    def get_latest_vertex_model(self, model_name: str):
        """If there are a few models with the same name, load the most recent one.
        It's highly recommended to keep only 1 model with the same name to avoid the confusion
        """
        self._vertex_init()

        # Search for models with the specified display name
        models = aiplatform.Model.list(filter=f"display_name={model_name}")

        if not models:
            return None  # Return None if no model with the given name exists

        # Sort models by update time in descending order to get the latest version
        latest_model = sorted(models, key=lambda x: x.update_time, reverse=True)[0]
        return latest_model

    def upload_artifacts_to_registry(self, file_list):
        # Initialize the GCS client
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        last_model = self.get_latest_vertex_model(self.model_name)
        if last_model:
            last_version = last_model.version_id
            new_version = int(last_version) + 1
        else:
            new_version = 1

        for file_name in file_list:
            # Upload the model file
            blob = bucket.blob(self.get_artifacts_path(new_version, file_name))
            blob.upload_from_filename(file_name)

        return self.create_model_version(new_version, last_model)

    # upload model to vertex ai model registry
    def upload(self):
        file_list = self.model_loader.dump()

        self.upload_artifacts_to_registry(file_list)

    def get_artifacts_path(self, version: int, file_name: str = None):
        dir_name = f"models/{self.model_name}/{version}"
        if file_name:
            return f"{dir_name}/{file_name}"

        return f"{dir_name}/"

    def create_model_version(self, version_number, last_version):
        artifact_uri = (
            f"gs://{self.bucket_name}/{self.get_artifacts_path(version_number)}"
        )

        if last_version:
            parent_model = f"projects/{self.project_name}/locations/{self.region}/models/{last_version.name}"
        else:
            parent_model = None

        release = aiplatform.Model.upload(
            display_name=self.model_name,
            artifact_uri=artifact_uri,
            parent_model=parent_model,
            serving_container_image_uri=self.serving_container_image_uri,
            serving_container_health_route=self.serving_container_health_route,
            serving_container_predict_route=self.serving_container_predict_route,
            sync=True,
            labels=self.labels,
        )

        return release

    def get_or_create_endpoint(self):
        endpoint = self.get_endpoint()
        if endpoint:
            return endpoint
        else:
            return self.create_endpoint()

    def get_endpoint(self):
        endpoints = aiplatform.Endpoint.list(
            filter=f"display_name={self.endpoint_name}", order_by="create_time"
        )
        if endpoints:
            return endpoints[0]

    def create_endpoint(self):
        return aiplatform.Endpoint.create(
            display_name=self.endpoint_name,
            project=self.project_name,
            location=self.region,
        )

    def deploy(self):
        self._vertex_init()

        endpoint = self.get_or_create_endpoint()
        model_version = self.upload()
        model_version.deploy(
            endpoint=endpoint,
            machine_type=self.machine_type,
            min_replica_count=self.min_replica_count,
            max_replica_count=self.max_replica_count,
            traffic_percentage=100,
        )

    def remote_predict(self, input_df: pd.DataFrame):
        self._vertex_init()

        if not self.endpoint:
            endpoint = self.get_endpoint()
            if not endpoint:
                raise Exception(
                    f"Endpoint '{self.endpoint_name}' not found. Please deploy the model first."
                )
            self.endpoint = endpoint

        instances = self.df_to_features(input_df)
        predictions = self.endpoint.predict(instances=instances)
        return predictions.predictions

    def local_predict(self, input_df: pd.DataFrame):
        if not self.model:
            self._vertex_init()
            self.load_model_from_registry()

        return self.model.predict(input_df)

    def local_predict_proba(self, input_df: pd.DataFrame):
        if not self.model:
            self._vertex_init()
            self.load_model_from_registry()
        return self.model.predict_proba(input_df)

    def load_model_from_registry(self):
        self.download_artifacts_from_registry()

        self.model = self.model_loader.load()

    def download_artifacts_from_registry(self):
        self._vertex_init()

        vertex_model = self.get_latest_vertex_model(self.model_name)

        if self.model_version:
            # reload the model using a specific model version
            vertex_model = aiplatform.Model(
                model_name=vertex_model.resource_name, version=str(self.model_version)
            )

        if not vertex_model:
            raise Exception(f"Model '{self.model_name}' not found in the registry.")

        artifact_uri = vertex_model.gca_resource.artifact_uri
        self.download_artifacts_from_uri(artifact_uri)

    def download_artifacts_from_uri(self, artifact_uri: str):
        storage_client = storage.Client()
        bucket_name, artifact_path = artifact_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=artifact_path)
        for blob in blobs:
            artifact_path = blob.name.split("/")[-1]
            blob.download_to_filename(artifact_path)

    def df_to_features(self, df: pd.DataFrame):
        return df.to_dict(orient="records")
