import pytest
import pandas as pd
import joblib
import os
from unittest.mock import patch, MagicMock
from orient_express import ModelExpress


@pytest.fixture
def sample_model():
    # Sample model fixture
    model = MagicMock()
    model.predict.return_value = [1, 0, 1]  # Mock prediction
    return model


@pytest.fixture
def model_express_instance(sample_model):
    return ModelExpress(
        model_name="test_model",
        project_name="test_project",
        bucket_name="test_bucket",
        model=sample_model,
        serialized_model_path="test_model.joblib",
    )


@pytest.fixture
def input_data():
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})


def test_get_latest_vertex_model(model_express_instance):
    with patch.object(model_express_instance, "_vertex_init"), patch(
        "google.cloud.aiplatform.Model.list",
        return_value=[MagicMock(update_time="2024-11-01T00:00:00Z")],
    ) as mock_list:

        latest_model = model_express_instance.get_latest_vertex_model("test_model")

        assert latest_model is not None
        mock_list.assert_called_once_with(filter="display_name=test_model")


def test_local_predict(model_express_instance, input_data):
    predictions = model_express_instance.local_predict(input_data)
    assert predictions == [1, 0, 1]


def test_get_artifacts_path(model_express_instance):
    path = model_express_instance.get_artifacts_path(1, "model.joblib")
    assert path == "models/test_model/1/model.joblib"


@pytest.mark.parametrize(
    "version, filename, expected_path",
    [
        (1, "model.joblib", "models/test_model/1/model.joblib"),
        (2, None, "models/test_model/2/"),
    ],
)
def test_get_artifacts_path_with_parametrize(
    model_express_instance, version, filename, expected_path
):
    path = model_express_instance.get_artifacts_path(version, file_name=filename)
    assert path == expected_path
