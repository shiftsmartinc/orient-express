from __future__ import annotations

import math
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar, overload

import joblib
import yaml
from google.cloud import aiplatform, storage
from google.cloud.storage import transfer_manager

from .utils.paths import get_cache_dir
from .utils.retry import get_gcs_retry_policy, get_vertex_retry_policy

if TYPE_CHECKING:
    from .predictors import Predictor

T = TypeVar("T")

ARTIFACT_DIR = get_cache_dir()
_last_vertex_init: tuple[str, str] | None = None


# The two policies this module applies, built by utils.retry so every module
# retries the same way. One per transport, held as a constant so each remote call
# inherits it rather than each site deciding for itself -- ad-hoc coverage is
# what let a transient 503 on Model.list fail a whole pipeline run.
#
# Both are predicate-based, so a genuinely missing model or a permission error
# fails immediately instead of burning the backoff budget, and the original
# exception reaches the caller -- the house decorator would re-raise a plain
# Exception and defeat get_vertex_model's own raise_exception contract.
VERTEX_RETRY = get_vertex_retry_policy()
GCS_RETRY = get_gcs_retry_policy()

# Files larger than this transfer as concurrent chunks; a typical ONNX
# artifact (~120 MB) is otherwise bottlenecked on a single stream.
CHUNKED_TRANSFER_THRESHOLD_BYTES = 8 * 1024 * 1024
TRANSFER_MAX_WORKERS = 8

# Per-HTTP-request timeout for model uploads, seconds. One request carries
# up to one 32MB chunk, and the workers above share the uplink, so on a slow
# connection each request legitimately takes minutes — google's 60s default
# (retried for at most 120s) aborts uploads that would have finished.
# Override per call with upload_timeout=...
DEFAULT_UPLOAD_TIMEOUT_SECONDS = 600.0


def _resolve_upload_timeout(timeout: float | None) -> float:
    if timeout is None:
        return DEFAULT_UPLOAD_TIMEOUT_SECONDS
    # 0/negative/inf/nan would otherwise reach the GCS client and fail in
    # confusing ways (or never time out at all)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            f"upload_timeout must be a positive finite number of seconds, "
            f"got {timeout!r}"
        )
    return float(timeout)


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
        self._endpoint_cache: dict[str, Any] = {}

    def deploy_to_endpoint(
        self,
        endpoint_name: str,
        machine_type: str,
        min_replica_count: int,
        max_replica_count: int,
    ):
        endpoint = self.get_or_create_endpoint(endpoint_name)
        VERTEX_RETRY(self.vertex_model.deploy)(
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
        # Endpoint.list is a full API round-trip; resolve each name once per
        # VertexModel so repeated remote_predict calls don't pay it again.
        cached = self._endpoint_cache.get(endpoint_name)
        if cached is not None:
            return cached
        endpoints = VERTEX_RETRY(aiplatform.Endpoint.list)(
            filter=f"display_name={endpoint_name}", order_by="create_time"
        )
        if endpoints:
            self._endpoint_cache[endpoint_name] = endpoints[0]
            return endpoints[0]

    def create_endpoint(self, endpoint_name: str):
        endpoint = VERTEX_RETRY(aiplatform.Endpoint.create)(
            display_name=endpoint_name,
            project=self.project_name,
            location=self.region,
        )
        self._endpoint_cache[endpoint_name] = endpoint
        return endpoint

    def remote_predict(
        self, endpoint_name: str, instances: list, parameters: dict | None = None
    ):
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            raise Exception(
                f"Endpoint '{endpoint_name}' not found. Please deploy the model first."
            )
        self.endpoint = endpoint
        # Retried too, not just the control plane: inference is idempotent, and
        # the predicate only matches transient statuses -- 429 in particular is
        # ordinary endpoint throttling, which backoff is the correct answer to.
        predictions = VERTEX_RETRY(self.endpoint.predict)(
            instances=instances, parameters=parameters
        )
        return predictions.predictions

    # the expected_type overload comes first: checkers match overloads in
    # order, and the general **kwargs overload would swallow expected_type=
    # calls and erase the narrowed return type
    @overload
    def get_local_predictor(
        self,
        device: str = "cpu",
        force_download: bool = False,
        *,
        expected_type: type[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def get_local_predictor(
        self, device: str = "cpu", force_download: bool = False, **kwargs: Any
    ) -> Any: ...

    def get_local_predictor(
        self,
        device: str = "cpu",
        force_download: bool = False,
        *,
        expected_type: type[T] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Download this model's artifacts (cached) and load a local predictor.

        Pass expected_type to narrow the static return type and get a runtime
        check that the artifact really is that predictor class:

            predictor = vertex_model.get_local_predictor(
                expected_type=BoundingBoxPredictor
            )

        Extra keyword arguments are forwarded to the predictor constructor
        (e.g. provider_options).
        """
        # Deferred: pulls onnxruntime/cv2, which vertex-only users never need
        from .predictors import get_predictor

        dir = os.path.join(ARTIFACT_DIR, self.model_name + "-" + str(self.version))
        self.download_artifacts(dir, force_download=force_download)
        if expected_type is None:
            return get_predictor(dir, device, **kwargs)
        return get_predictor(dir, device, expected_type=expected_type, **kwargs)

    def download_artifacts(self, dir: str, force_download: bool = True):
        download_artifacts(
            dir, self.vertex_model.gca_resource.artifact_uri, force_download
        )


def vertex_init(project_name: str, region: str):
    global _last_vertex_init
    if _last_vertex_init is not None and _last_vertex_init != (project_name, region):
        warnings.warn(
            f"Vertex AI SDK re-initialized with project '{project_name}' region "
            f"'{region}' (was project '{_last_vertex_init[0]}' region "
            f"'{_last_vertex_init[1]}'). The SDK holds this state globally: models "
            "and endpoints obtained before this call remain bound to the previous "
            "project/region.",
            stacklevel=2,
        )
    VERTEX_RETRY(aiplatform.init)(project=project_name, location=region)
    _last_vertex_init = (project_name, region)


# Retry the whole blob download rather than passing a retry policy in:
# download_chunks_concurrently accepts no `retry` argument, so the >8MB branch --
# which is every ONNX artifact we ship -- would otherwise be the one unprotected
# path while the small-file branch inherits the client's DEFAULT_RETRY. Wrapping
# means a failed chunked transfer is re-attempted as a whole.
@GCS_RETRY
def _download_blob(blob, download_path: str):
    if blob.size and blob.size > CHUNKED_TRANSFER_THRESHOLD_BYTES:
        transfer_manager.download_chunks_concurrently(
            blob,
            download_path,
            worker_type=transfer_manager.THREAD,
            max_workers=TRANSFER_MAX_WORKERS,
        )
    else:
        blob.download_to_filename(download_path)


def _upload_file(bucket, file_path: str, blob_name: str, timeout: float):
    # Same GCS policy as everywhere else, but the total-time budget must scale
    # with the request timeout: the default gives up after 120s, so with a
    # generous per-request timeout a single transient failure late in a slow
    # chunk would kill the upload without ever retrying.
    retry_policy = get_gcs_retry_policy(timeout=max(timeout * 2, 120.0))
    blob = bucket.blob(blob_name)
    if os.path.getsize(file_path) > CHUNKED_TRANSFER_THRESHOLD_BYTES:
        transfer_manager.upload_chunks_concurrently(
            file_path,
            blob,
            worker_type=transfer_manager.THREAD,
            max_workers=TRANSFER_MAX_WORKERS,
            timeout=timeout,
            retry=retry_policy,
        )
    else:
        blob.upload_from_filename(file_path, timeout=timeout, retry=retry_policy)


def download_artifacts(dir: str, artifact_uri: str, force_download: bool = True):
    storage_client = storage.Client()
    bucket_name, artifact_path = artifact_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    os.makedirs(dir, exist_ok=True)
    prefix = artifact_path.rstrip("/") + "/"
    to_download = []
    # list_blobs paginates lazily, so the retry has to wrap the listing itself
    # rather than the loop -- a transient failure fetching page 2 would otherwise
    # surface mid-iteration.
    for blob in bucket.list_blobs(prefix=artifact_path, retry=GCS_RETRY):
        if blob.name.startswith(prefix):
            relative_path = blob.name[len(prefix) :]
        else:
            relative_path = blob.name.split("/")[-1]
        if not relative_path:  # directory placeholder object
            continue
        # blob names are always '/'-separated; split so nested paths get
        # native separators instead of a mixed 'dir\\sub/file' on Windows
        download_path = os.path.join(dir, *relative_path.split("/"))
        if not force_download and os.path.exists(download_path):
            continue
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        to_download.append((blob, download_path))
    with ThreadPoolExecutor(max_workers=TRANSFER_MAX_WORKERS) as pool:
        futures = [
            pool.submit(_download_blob, blob, path) for blob, path in to_download
        ]
        for future in futures:
            future.result()


def upload_model(
    model: Predictor,
    model_name: str,
    project_name: str,
    region: str,
    bucket_name: str,
    serving_container_image_uri: str = "",
    serving_container_health_route: str = "",
    serving_container_predict_route: str = "",
    labels: dict[str, str] | None = None,
    upload_timeout: float | None = None,
):
    """Upload a Predictor model to Vertex AI Model Registry.

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
        upload_timeout: Per-HTTP-request timeout in seconds for the artifact
            transfer (each request carries up to one 32MB chunk). Default
            DEFAULT_UPLOAD_TIMEOUT_SECONDS; raise it on slow connections.

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
            upload_timeout,
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
    labels: dict[str, str] | None = None,
    upload_timeout: float | None = None,
):
    """Upload a joblib-serializable model to Vertex AI Model Registry.

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
        upload_timeout: Per-HTTP-request timeout in seconds for the artifact
            transfer (see upload_model)

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
            upload_timeout,
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
    labels: dict[str, str] | None = None,
    upload_timeout: float | None = None,
) -> VertexModel:
    parent_model = get_vertex_model(
        model_name, project_name, region, raise_exception=False
    )
    if parent_model:
        # Next version = highest existing version + 1, matching how the registry
        # assigns version ids. get_vertex_model returns the *default* version,
        # which can be older than the newest one, so incrementing from it can
        # collide with an existing artifact folder.
        registry = aiplatform.models.ModelRegistry(
            model=parent_model.vertex_model.resource_name
        )
        versions = VERTEX_RETRY(registry.list_versions)()
        version = max(int(v.version_id) for v in versions) + 1
    else:
        version = 1

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    timeout = _resolve_upload_timeout(upload_timeout)
    artifact_dir = f"models/{model_name}/{version}/"
    if next(iter(bucket.list_blobs(prefix=artifact_dir, max_results=1)), None):
        raise FileExistsError(
            f"gs://{bucket_name}/{artifact_dir} already contains files; "
            "refusing to overwrite an existing model artifact"
        )
    with ThreadPoolExecutor(max_workers=TRANSFER_MAX_WORKERS) as pool:
        futures = [
            pool.submit(
                _upload_file,
                bucket,
                file_name,
                f"{artifact_dir}{os.path.basename(file_name)}",
                timeout,
            )
            for file_name in file_list
        ]
        for future in futures:
            future.result()

    artifact_uri = f"gs://{bucket_name}/{artifact_dir}"

    if parent_model:
        parent_model_uri = (
            f"projects/{project_name}/locations/{region}/models/{parent_model.name}"
        )
    else:
        parent_model_uri = None

    if labels is None:
        labels = {}

    release = VERTEX_RETRY(aiplatform.Model.upload)(
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
    # The call this issue was filed for: a transient 503 here used to abort the
    # caller outright. Note the raise below is NOT retried -- a model that is
    # genuinely absent must fail fast rather than spend the backoff budget.
    models = VERTEX_RETRY(aiplatform.Model.list)(filter=f"display_name={model_name}")
    if not models:
        if raise_exception:
            raise Exception(
                f"Model '{model_name}' not found in registry for project '{project_name}' region '{region}'"
            )
        else:
            return None
    if len(models) > 1:
        warnings.warn(
            f"Multiple models found with name '{model_name}'. Using the latest one.",
            stacklevel=2,
        )

    latest_model = sorted(models, key=lambda x: x.update_time, reverse=True)[0]
    resource_name = latest_model.resource_name

    if version is None:
        return VertexModel(
            latest_model, model_name, project_name, region, int(latest_model.version_id)
        )

    try:
        model = VERTEX_RETRY(aiplatform.Model)(
            model_name=resource_name, version=str(version)
        )
    except Exception as e:
        if raise_exception:
            raise Exception(
                f"Failed to fetch model '{model_name}' with version '{version}' in registry for project '{project_name}' region '{region}'"
            ) from e
        return None
    return VertexModel(model, model_name, project_name, region, version)
