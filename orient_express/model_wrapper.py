import os
import joblib
import logging
import pandas as pd
from google.cloud import storage
from google.cloud import aiplatform


class ModelExpress:
    def __init__(
        self,
        model_name,
        project_name,
        bucket_name,
        model_version=None,
        model=None,
        region="us-central1",
        serialized_model_path="model.joblib",
        docker_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
        endpoint_name=None,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1,
    ):
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.region = region
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.serialized_model_path = serialized_model_path
        self.docker_image_uri = docker_image_uri
        self.machine_type = machine_type
        self.min_replica_count = min_replica_count
        self.max_replica_count = max_replica_count
        self.endpoint = None

        if not endpoint_name:
            self.endpoint_name = f"model-xpress-{model_name}"
        else:
            self.endpoint_name = endpoint_name

    def colab_auth(self):
        from google.colab import auth

        auth.authenticate_user()

    def _vertex_init(self):
        aiplatform.init(project=self.project_name, location=self.region)

    def get_latest_vertex_model(self, model_name):
        self._vertex_init()

        # Search for models with the specified display name
        models = aiplatform.Model.list(filter=f"display_name={model_name}")

        if not models:
            return None  # Return None if no model with the given name exists

        # Sort models by update time in descending order to get the latest version
        latest_model = sorted(models, key=lambda x: x.update_time, reverse=True)[0]
        return latest_model

    # upload model to vertex ai model registry
    def upload(self):
        joblib.dump(self.model, self.serialized_model_path)
        logging.info(f"Model saved to {self.serialized_model_path}")

        # Initialize the GCS client
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        last_model = self.get_latest_vertex_model(self.model_name)
        if last_model:
            last_version = last_model.version_id
            new_version = int(last_version) + 1
        else:
            new_version = 1

        # Upload the model file
        blob = bucket.blob(
            self.get_artifacts_path(new_version, self.serialized_model_path)
        )
        blob.upload_from_filename(self.serialized_model_path)

        return self.create_model_version(new_version, last_model)

    def get_artifacts_path(self, version, file_name=None):
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
            serving_container_image_uri=self.docker_image_uri,
            sync=True,
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

    def remote_predict(self, input_df):
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

    def local_predict(self, input_df):
        if not self.model:
            self.load_model_from_registry()

        return self.model.predict(input_df)

    def load_model_from_registry(self):
        if self.model_version:
            vertex_model = aiplatform.Model(
                model_name=self.model_name, version=self.model_version
            )

        else:
            vertex_model = self.get_latest_vertex_model(self.model_name)

        if not vertex_model:
            raise Exception(f"Model '{self.model_name}' not found in the registry.")

        artifact_uri = vertex_model.gca_resource.artifact_uri
        self.download_artifacts(artifact_uri)

        self.model = joblib.load(self.serialized_model_path)

    def download_artifacts(self, artifact_uri):
        storage_client = storage.Client()
        bucket_name, artifact_path = artifact_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=artifact_path)
        for blob in blobs:
            artifact_path = blob.name.split("/")[-1]
            blob.download_to_filename(artifact_path)

    def df_to_features(self, df: pd.DataFrame):
        #
        return df.to_dict(orient="records")
