"""
Tests for vertex.py functionality.

These tests verify:
1. Artifact path construction for uploads and downloads
2. Versioning logic (increment when parent exists, start at 1 otherwise)
3. get_vertex_model filtering and sorting
4. Endpoint management (get_or_create_endpoint, deploy_to_endpoint)

All GCS and Vertex AI SDK calls are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import tempfile
import os

# Import the module for accessing globals like _vertex_initialized
import orient_express.vertex as vertex_module

from orient_express.vertex import (
    upload_model,
    upload_model_joblib,
    download_artifacts,
    get_vertex_model,
    vertex_init,
    VertexModel,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_storage_client():
    """Creates a mock GCS storage client."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return mock_client, mock_bucket


@pytest.fixture
def mock_aiplatform():
    """Creates mock aiplatform module components."""
    mock_model_class = MagicMock()
    mock_endpoint_class = MagicMock()
    return mock_model_class, mock_endpoint_class


@pytest.fixture
def mock_vertex_model_factory():
    """Creates a mock Vertex AI Model object."""

    def _create(
        name="model-123",
        version_id="1",
        update_time=None,
        artifact_uri="gs://bucket/models/test/1/",
    ):
        mock = MagicMock()
        mock.name = name
        mock.version_id = version_id
        mock.update_time = update_time or datetime(2024, 1, 1, 12, 0, 0)
        mock.gca_resource.artifact_uri = artifact_uri
        return mock

    return _create


@pytest.fixture
def mock_predictor():
    """Creates a mock Predictor that can be uploaded."""
    mock = MagicMock()
    mock.get_serving_container_image_uri.return_value = "gcr.io/test/image:v1"
    mock.get_serving_container_health_route.return_value = "/v1/models/test"
    mock.get_serving_container_predict_route.return_value = "/v1/models/test:predict"

    def dump_side_effect(dir):
        # Create fake files that dump would create
        metadata_path = os.path.join(dir, "metadata.yaml")
        model_path = os.path.join(dir, "model.onnx")
        with open(metadata_path, "w") as f:
            f.write("model_type: test\n")
        with open(model_path, "w") as f:
            f.write("fake onnx content")
        return [metadata_path, model_path]

    mock.dump.side_effect = dump_side_effect
    return mock


# -----------------------------------------------------------------------------
# Artifact Path Construction Tests
# -----------------------------------------------------------------------------


class TestArtifactPathConstruction:
    """Tests for artifact path construction in upload and download."""

    def test_upload_constructs_correct_blob_paths(
        self, mock_storage_client, mock_predictor
    ):
        """Uploaded files go to models/{name}/{version}/{filename}."""
        mock_client, mock_bucket = mock_storage_client
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            # No existing model
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(
                name="new-model", version_id="1"
            )

            upload_model(
                model=mock_predictor,
                model_name="my-detector",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
            )

            # Verify blob paths
            blob_calls = mock_bucket.blob.call_args_list
            blob_paths = [call[0][0] for call in blob_calls]

            # Should have paths like models/my-detector/1/metadata.yaml
            assert any("models/my-detector/1/" in path for path in blob_paths)
            assert any("metadata.yaml" in path for path in blob_paths)
            assert any("model.onnx" in path for path in blob_paths)

    def test_upload_version_increments_path(self, mock_storage_client, mock_predictor):
        """When parent model exists, version in path increments."""
        mock_client, mock_bucket = mock_storage_client
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        # Create existing model with version 3
        existing_model = MagicMock()
        existing_model.name = "existing-model-id"
        existing_model.version_id = "3"
        existing_model.update_time = datetime(2024, 1, 1)
        existing_model.gca_resource.artifact_uri = "gs://bucket/models/my-detector/3/"

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = [existing_model]
            mock_model_class.upload.return_value = MagicMock(
                name="new-model", version_id="4"
            )

            upload_model(
                model=mock_predictor,
                model_name="my-detector",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
            )

            # Verify blob paths use version 4
            blob_calls = mock_bucket.blob.call_args_list
            blob_paths = [call[0][0] for call in blob_calls]

            assert any("models/my-detector/4/" in path for path in blob_paths)
            assert not any("models/my-detector/3/" in path for path in blob_paths)

    def test_download_artifacts_uses_correct_prefix(self, mock_storage_client):
        """download_artifacts lists blobs with correct prefix from artifact_uri."""
        mock_client, mock_bucket = mock_storage_client

        mock_blob1 = MagicMock()
        mock_blob1.name = "models/test-model/2/metadata.yaml"
        mock_blob2 = MagicMock()
        mock_blob2.name = "models/test-model/2/model.onnx"
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            download_artifacts(tmpdir, "gs://test-bucket/models/test-model/2/")

            # Verify list_blobs called with correct prefix
            mock_bucket.list_blobs.assert_called_once_with(
                prefix="models/test-model/2/"
            )

    def test_download_artifacts_saves_to_correct_paths(self, mock_storage_client):
        """Downloaded files are saved to the target directory with correct names."""
        mock_client, mock_bucket = mock_storage_client

        mock_blob1 = MagicMock()
        mock_blob1.name = "models/test-model/2/metadata.yaml"
        mock_blob2 = MagicMock()
        mock_blob2.name = "models/test-model/2/model.onnx"
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            download_artifacts(tmpdir, "gs://test-bucket/models/test-model/2/")

            # Verify download_to_filename called with correct paths
            mock_blob1.download_to_filename.assert_called_once_with(
                os.path.join(tmpdir, "metadata.yaml")
            )
            mock_blob2.download_to_filename.assert_called_once_with(
                os.path.join(tmpdir, "model.onnx")
            )


# -----------------------------------------------------------------------------
# Versioning Logic Tests
# -----------------------------------------------------------------------------


class TestVersioningLogic:
    """Tests for model versioning in upload_model_with_files."""

    def test_first_upload_sets_version_1(self, mock_storage_client, mock_predictor):
        """When no existing model, version is set to 1."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_uploaded = MagicMock(name="new-model", version_id="1")
            mock_model_class.upload.return_value = mock_uploaded

            result = upload_model(
                model=mock_predictor,
                model_name="new-model",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
            )

            # Verify upload called with parent_model=None
            upload_call = mock_model_class.upload.call_args
            assert upload_call.kwargs["parent_model"] is None

            # Verify returned VertexModel has version 1
            assert result.version == 1

    def test_subsequent_upload_increments_version(
        self, mock_storage_client, mock_predictor, mock_vertex_model_factory
    ):
        """When parent model exists, version increments from latest."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        existing = mock_vertex_model_factory(name="existing-id", version_id="5")

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = [existing]
            mock_uploaded = MagicMock(name="new-version", version_id="6")
            mock_model_class.upload.return_value = mock_uploaded

            result = upload_model(
                model=mock_predictor,
                model_name="my-model",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
            )

            # Verify upload called with correct parent_model URI
            upload_call = mock_model_class.upload.call_args
            assert (
                upload_call.kwargs["parent_model"]
                == "projects/test-project/locations/us-central1/models/existing-id"
            )

            # Verify returned VertexModel has incremented version
            assert result.version == 6

    def test_version_uses_latest_when_multiple_exist(
        self, mock_storage_client, mock_predictor, mock_vertex_model_factory
    ):
        """When multiple versions exist, increments from the highest version."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        # Multiple existing versions, not in order
        v1 = mock_vertex_model_factory(
            name="model-v1",
            version_id="1",
            update_time=datetime(2024, 1, 1),
        )
        v3 = mock_vertex_model_factory(
            name="model-v3",
            version_id="3",
            update_time=datetime(2024, 3, 1),
        )
        v2 = mock_vertex_model_factory(
            name="model-v2",
            version_id="2",
            update_time=datetime(2024, 2, 1),
        )

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            # get_vertex_model sorts by update_time to find latest
            mock_model_class.list.return_value = [v1, v3, v2]
            mock_uploaded = MagicMock(name="new-version", version_id="4")
            mock_model_class.upload.return_value = mock_uploaded

            result = upload_model(
                model=mock_predictor,
                model_name="my-model",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
            )

            # Should use v3 (latest by update_time) as parent
            # Version should be 3 + 1 = 4
            assert result.version == 4


# -----------------------------------------------------------------------------
# get_vertex_model Tests
# -----------------------------------------------------------------------------


class TestGetVertexModel:
    """Tests for get_vertex_model filtering and sorting."""

    def test_returns_latest_by_update_time_when_no_version_specified(
        self, mock_vertex_model_factory
    ):
        """Without version parameter, returns model with most recent update_time."""
        v1 = mock_vertex_model_factory(
            name="model-v1",
            version_id="1",
            update_time=datetime(2024, 1, 1, 10, 0, 0),
        )
        v2 = mock_vertex_model_factory(
            name="model-v2",
            version_id="2",
            update_time=datetime(2024, 1, 15, 10, 0, 0),
        )
        v3 = mock_vertex_model_factory(
            name="model-v3",
            version_id="3",
            update_time=datetime(2024, 1, 10, 10, 0, 0),
        )

        with (
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = [v1, v2, v3]

            result = get_vertex_model(
                model_name="my-model",
                project_name="test-project",
                region="us-central1",
                version=None,
            )

            # v2 has the latest update_time
            assert result.version == 2
            assert result.vertex_model.name == "model-v2"

    def test_returns_specific_version_when_specified(self, mock_vertex_model_factory):
        """With version parameter, returns the matching version."""
        v1 = mock_vertex_model_factory(name="model-v1", version_id="1")
        v2 = mock_vertex_model_factory(name="model-v2", version_id="2")
        v3 = mock_vertex_model_factory(name="model-v3", version_id="3")

        with (
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = [v1, v2, v3]

            result = get_vertex_model(
                model_name="my-model",
                project_name="test-project",
                region="us-central1",
                version=2,
            )

            assert result.version == 2
            assert result.vertex_model.name == "model-v2"

    def test_raises_when_model_not_found_and_raise_exception_true(self):
        """Raises exception when no models found and raise_exception=True."""
        with (
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []

            with pytest.raises(Exception) as exc_info:
                get_vertex_model(
                    model_name="nonexistent",
                    project_name="test-project",
                    region="us-central1",
                    raise_exception=True,
                )

            assert "not found" in str(exc_info.value).lower()

    def test_returns_none_when_model_not_found_and_raise_exception_false(self):
        """Returns None when no models found and raise_exception=False."""
        with (
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []

            result = get_vertex_model(
                model_name="nonexistent",
                project_name="test-project",
                region="us-central1",
                raise_exception=False,
            )

            assert result is None

    def test_raises_when_specific_version_not_found(self, mock_vertex_model_factory):
        """Raises exception when requested version doesn't exist."""
        v1 = mock_vertex_model_factory(name="model-v1", version_id="1")
        v2 = mock_vertex_model_factory(name="model-v2", version_id="2")

        with (
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = [v1, v2]

            with pytest.raises(Exception) as exc_info:
                get_vertex_model(
                    model_name="my-model",
                    project_name="test-project",
                    region="us-central1",
                    version=99,
                    raise_exception=True,
                )

            assert "version" in str(exc_info.value).lower()

    def test_returns_none_when_specific_version_not_found_and_raise_false(
        self, mock_vertex_model_factory
    ):
        """Returns None when version not found and raise_exception=False."""
        v1 = mock_vertex_model_factory(name="model-v1", version_id="1")

        with (
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = [v1]

            result = get_vertex_model(
                model_name="my-model",
                project_name="test-project",
                region="us-central1",
                version=99,
                raise_exception=False,
            )

            assert result is None

    def test_uses_display_name_filter_in_list(self, mock_vertex_model_factory):
        """Model.list is called with correct display_name filter."""
        with (
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []

            get_vertex_model(
                model_name="specific-model-name",
                project_name="test-project",
                region="us-central1",
                raise_exception=False,
            )

            mock_model_class.list.assert_called_once_with(
                filter="display_name=specific-model-name"
            )


# -----------------------------------------------------------------------------
# Endpoint Management Tests
# -----------------------------------------------------------------------------


class TestEndpointManagement:
    """Tests for endpoint creation, retrieval, and deployment."""

    def test_get_endpoint_returns_existing(self):
        """get_endpoint returns first endpoint when list finds matches."""
        mock_endpoint = MagicMock()
        mock_endpoint.display_name = "my-endpoint"

        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.list.return_value = [mock_endpoint]

            vertex_model = VertexModel(
                vertex_model=MagicMock(),
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            result = vertex_model.get_endpoint("my-endpoint")

            assert result == mock_endpoint
            mock_endpoint_class.list.assert_called_once_with(
                filter="display_name=my-endpoint", order_by="create_time"
            )

    def test_get_endpoint_returns_none_when_not_found(self):
        """get_endpoint returns None when no endpoints match."""
        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.list.return_value = []

            vertex_model = VertexModel(
                vertex_model=MagicMock(),
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            result = vertex_model.get_endpoint("nonexistent")

            assert result is None

    def test_create_endpoint_calls_endpoint_create(self):
        """create_endpoint calls Endpoint.create with correct parameters."""
        mock_new_endpoint = MagicMock()

        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.create.return_value = mock_new_endpoint

            vertex_model = VertexModel(
                vertex_model=MagicMock(),
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            result = vertex_model.create_endpoint("new-endpoint")

            assert result == mock_new_endpoint
            mock_endpoint_class.create.assert_called_once_with(
                display_name="new-endpoint",
                project="test-project",
                location="us-central1",
            )

    def test_get_or_create_endpoint_returns_existing_when_found(self):
        """get_or_create_endpoint returns existing endpoint without creating."""
        mock_existing = MagicMock()

        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.list.return_value = [mock_existing]

            vertex_model = VertexModel(
                vertex_model=MagicMock(),
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            result = vertex_model.get_or_create_endpoint("existing-endpoint")

            assert result == mock_existing
            mock_endpoint_class.create.assert_not_called()

    def test_get_or_create_endpoint_creates_when_not_found(self):
        """get_or_create_endpoint creates new endpoint when none exists."""
        mock_new = MagicMock()

        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.list.return_value = []
            mock_endpoint_class.create.return_value = mock_new

            vertex_model = VertexModel(
                vertex_model=MagicMock(),
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            result = vertex_model.get_or_create_endpoint("new-endpoint")

            assert result == mock_new
            mock_endpoint_class.create.assert_called_once()

    def test_deploy_to_endpoint_calls_deploy_with_correct_args(self):
        """deploy_to_endpoint passes all parameters to vertex_model.deploy."""
        mock_inner_model = MagicMock()
        mock_endpoint = MagicMock()

        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.list.return_value = [mock_endpoint]

            vertex_model = VertexModel(
                vertex_model=mock_inner_model,
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            vertex_model.deploy_to_endpoint(
                endpoint_name="my-endpoint",
                machine_type="n1-standard-4",
                min_replica_count=1,
                max_replica_count=3,
            )

            mock_inner_model.deploy.assert_called_once_with(
                endpoint=mock_endpoint,
                machine_type="n1-standard-4",
                min_replica_count=1,
                max_replica_count=3,
                traffic_percentage=100,
            )

    def test_remote_predict_raises_when_endpoint_not_found(self):
        """remote_predict raises exception if endpoint doesn't exist."""
        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.list.return_value = []

            vertex_model = VertexModel(
                vertex_model=MagicMock(),
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            with pytest.raises(Exception) as exc_info:
                vertex_model.remote_predict(
                    instances=[{"data": "test"}], endpoint_name="nonexistent"
                )

            assert "not found" in str(exc_info.value).lower()

    def test_remote_predict_calls_endpoint_predict(self):
        """remote_predict calls endpoint.predict and returns predictions."""
        mock_endpoint = MagicMock()
        mock_endpoint.predict.return_value = MagicMock(
            predictions=[{"class": "cat", "score": 0.9}]
        )

        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.list.return_value = [mock_endpoint]

            vertex_model = VertexModel(
                vertex_model=MagicMock(),
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            instances = [{"image": "base64data"}]
            result = vertex_model.remote_predict(
                instances=instances, endpoint_name="my-endpoint"
            )

            mock_endpoint.predict.assert_called_once_with(
                instances=instances, parameters=None
            )
            assert result == [{"class": "cat", "score": 0.9}]

    def test_remote_predict_calls_endpoint_predict_with_parameters(self):
        mock_endpoint = MagicMock()
        mock_endpoint.predict.return_value = MagicMock(
            predictions=[{"class": "cat", "score": 0.9}, {"class": "dog", "score": 0.8}]
        )

        with patch("orient_express.vertex.aiplatform.Endpoint") as mock_endpoint_class:
            mock_endpoint_class.list.return_value = [mock_endpoint]

            vertex_model = VertexModel(
                vertex_model=MagicMock(),
                model_name="test",
                project_name="test-project",
                region="us-central1",
                version=1,
            )
            instances = [{"image": "base64data"}, {"image": "base64data2"}]
            result = vertex_model.remote_predict(
                instances=instances,
                endpoint_name="my-endpoint",
                parameters={"foo": "bar"},
            )

            mock_endpoint.predict.assert_called_once_with(
                instances=instances, parameters={"foo": "bar"}
            )
            assert result == [
                {"class": "cat", "score": 0.9},
                {"class": "dog", "score": 0.8},
            ]


# -----------------------------------------------------------------------------
# vertex_init Tests
# -----------------------------------------------------------------------------


class TestVertexInit:
    """Tests for vertex_init initialization."""

    def test_initializes_aiplatform_once(self):
        """vertex_init only calls aiplatform.init once."""
        # Reset the global state
        vertex_module._vertex_initialized = False

        with patch("orient_express.vertex.aiplatform.init") as mock_init:
            vertex_init("project1", "us-central1")
            vertex_init("project1", "us-central1")
            vertex_init("project1", "us-central1")

            # Should only be called once despite multiple calls
            mock_init.assert_called_once_with(
                project="project1", location="us-central1"
            )


# -----------------------------------------------------------------------------
# VertexModel.get_local_predictor Tests
# -----------------------------------------------------------------------------


class TestGetLocalPredictor:
    """Tests for VertexModel.get_local_predictor."""

    def test_downloads_artifacts_and_loads_predictor(self, mock_storage_client):
        """get_local_predictor downloads artifacts then loads predictor."""
        mock_client, mock_bucket = mock_storage_client

        mock_blob = MagicMock()
        mock_blob.name = "models/test/1/metadata.yaml"
        mock_bucket.list_blobs.return_value = [mock_blob]

        mock_inner_model = MagicMock()
        mock_inner_model.gca_resource.artifact_uri = "gs://bucket/models/test/1/"

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.get_predictor") as mock_get_predictor,
        ):
            mock_get_predictor.return_value = MagicMock()

            vertex_model = VertexModel(
                vertex_model=mock_inner_model,
                model_name="test-model",
                project_name="test-project",
                region="us-central1",
                version=1,
            )

            result = vertex_model.get_local_predictor()

            # Verify get_predictor was called
            mock_get_predictor.assert_called_once()
            # The directory should contain model name and version
            call_arg = mock_get_predictor.call_args[0][0]
            assert "test-model" in call_arg
            assert "1" in call_arg


# -----------------------------------------------------------------------------
# upload_model Tests
# -----------------------------------------------------------------------------


class TestUploadModel:
    """Tests for the upload_model function."""

    def test_uses_predictor_container_routes_by_default(
        self, mock_storage_client, mock_predictor
    ):
        """upload_model uses predictor's container routes when not specified."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            upload_model(
                model=mock_predictor,
                model_name="test",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
            )

            upload_call = mock_model_class.upload.call_args
            assert (
                upload_call.kwargs["serving_container_image_uri"]
                == "gcr.io/test/image:v1"
            )
            assert (
                upload_call.kwargs["serving_container_health_route"]
                == "/v1/models/test"
            )
            assert (
                upload_call.kwargs["serving_container_predict_route"]
                == "/v1/models/test:predict"
            )

    def test_uses_override_container_routes_when_specified(
        self, mock_storage_client, mock_predictor
    ):
        """upload_model uses provided container routes over predictor defaults."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            upload_model(
                model=mock_predictor,
                model_name="test",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
                serving_container_image_uri="custom-image:v2",
                serving_container_health_route="/custom/health",
                serving_container_predict_route="/custom/predict",
            )

            upload_call = mock_model_class.upload.call_args
            assert (
                upload_call.kwargs["serving_container_image_uri"] == "custom-image:v2"
            )
            assert (
                upload_call.kwargs["serving_container_health_route"] == "/custom/health"
            )
            assert (
                upload_call.kwargs["serving_container_predict_route"]
                == "/custom/predict"
            )

    def test_passes_labels_to_upload(self, mock_storage_client, mock_predictor):
        """upload_model passes labels to Model.upload."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            upload_model(
                model=mock_predictor,
                model_name="test",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
                labels={"env": "prod", "team": "ml"},
            )

            upload_call = mock_model_class.upload.call_args
            assert upload_call.kwargs["labels"] == {"env": "prod", "team": "ml"}

    def test_sets_model_name_environment_variable(
        self, mock_storage_client, mock_predictor
    ):
        """upload_model sets MODEL_NAME in container environment variables."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            upload_model(
                model=mock_predictor,
                model_name="my-detector",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
            )

            upload_call = mock_model_class.upload.call_args
            assert upload_call.kwargs["serving_container_environment_variables"] == {
                "MODEL_NAME": "my-detector"
            }


class TestUploadModelJoblib:
    """Tests for the upload_model_joblib function."""

    def test_serializes_model_with_joblib(self, mock_storage_client):
        """upload_model_joblib saves model using joblib."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        # Create a simple model that can be serialized
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
            patch("orient_express.vertex.joblib.dump") as mock_joblib_dump,
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            upload_model_joblib(
                model=model,
                model_name="test-joblib",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
                serving_container_image_uri="custom-image:v1",
                serving_container_health_route="/health",
                serving_container_predict_route="/predict",
            )

            # Verify joblib.dump was called with the model
            mock_joblib_dump.assert_called_once()
            assert mock_joblib_dump.call_args[0][0] is model

    def test_creates_metadata_with_joblib_type(self, mock_storage_client):
        """upload_model_joblib creates metadata.yaml with model_type 'joblib'."""
        mock_client, mock_bucket = mock_storage_client
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            upload_model_joblib(
                model=model,
                model_name="test-joblib",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
                serving_container_image_uri="custom-image:v1",
                serving_container_health_route="/health",
                serving_container_predict_route="/predict",
            )

            # Verify blob paths include metadata.yaml and model.joblib
            blob_calls = mock_bucket.blob.call_args_list
            blob_paths = [call[0][0] for call in blob_calls]

            assert any("metadata.yaml" in path for path in blob_paths)
            assert any("model.joblib" in path for path in blob_paths)

    def test_passes_container_routes_to_upload(self, mock_storage_client):
        """upload_model_joblib passes container routes to Model.upload."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            upload_model_joblib(
                model=model,
                model_name="test-joblib",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
                serving_container_image_uri="my-container:v2",
                serving_container_health_route="/custom/health",
                serving_container_predict_route="/custom/predict",
            )

            upload_call = mock_model_class.upload.call_args
            assert (
                upload_call.kwargs["serving_container_image_uri"] == "my-container:v2"
            )
            assert (
                upload_call.kwargs["serving_container_health_route"] == "/custom/health"
            )
            assert (
                upload_call.kwargs["serving_container_predict_route"]
                == "/custom/predict"
            )

    def test_passes_labels_to_upload(self, mock_storage_client):
        """upload_model_joblib passes labels to Model.upload."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            upload_model_joblib(
                model=model,
                model_name="test-joblib",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
                serving_container_image_uri="custom-image:v1",
                serving_container_health_route="/health",
                serving_container_predict_route="/predict",
                labels={"env": "staging", "owner": "data-team"},
            )

            upload_call = mock_model_class.upload.call_args
            assert upload_call.kwargs["labels"] == {
                "env": "staging",
                "owner": "data-team",
            }

    def test_returns_vertex_model(self, mock_storage_client):
        """upload_model_joblib returns a VertexModel instance."""
        mock_client, mock_bucket = mock_storage_client
        mock_bucket.blob.return_value = MagicMock()

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()

        with (
            patch("orient_express.vertex.storage.Client", return_value=mock_client),
            patch("orient_express.vertex.aiplatform.Model") as mock_model_class,
            patch("orient_express.vertex.aiplatform.init"),
        ):
            mock_model_class.list.return_value = []
            mock_model_class.upload.return_value = MagicMock(name="new", version_id="1")

            result = upload_model_joblib(
                model=model,
                model_name="test-joblib",
                project_name="test-project",
                region="us-central1",
                bucket_name="test-bucket",
                serving_container_image_uri="custom-image:v1",
                serving_container_health_route="/health",
                serving_container_predict_route="/predict",
            )

            assert isinstance(result, VertexModel)
            assert result.model_name == "test-joblib"
            assert result.version == 1
