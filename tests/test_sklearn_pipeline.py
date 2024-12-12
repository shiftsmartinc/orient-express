import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from orient_express.sklearn_pipeline import (
    LabelEncoderTransformer,
)  # Replace with your actual module name


@pytest.fixture
def sample_data(default_seed):
    X, y = make_classification(
        n_samples=50,
        n_features=4,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=default_seed,
    )
    return X, np.array([str(f"label-{item}") for item in y])


@pytest.fixture
def label_encoder(sample_data):
    X, y_labels = sample_data

    # Generate synthetic data for testing
    label_encoder = LabelEncoder()
    label_encoder.fit(y_labels)

    return label_encoder


@pytest.fixture
def y_as_ids(label_encoder, sample_data):
    X, y_labels = sample_data
    return label_encoder.transform(y_labels)


@pytest.fixture
def transformer(label_encoder, default_seed):
    model = RandomForestClassifier(random_state=default_seed)
    return LabelEncoderTransformer(model=model, label_encoder=label_encoder)


def test_fit(transformer, sample_data, y_as_ids):
    X, y_labels = sample_data

    # Fit the transformer
    transformer.fit(X, y_as_ids)

    # Assert that the model is fitted (e.g., has feature importances)
    assert hasattr(transformer.model, "feature_importances_")


@pytest.fixture
def expected_labels():
    return np.array(
        [
            "label-1",
            "label-1",
            "label-1",
            "label-0",
            "label-0",
            "label-1",
            "label-1",
            "label-0",
            "label-0",
            "label-0",
            "label-1",
            "label-1",
            "label-0",
            "label-0",
            "label-0",
            "label-1",
            "label-0",
            "label-1",
            "label-1",
            "label-1",
            "label-0",
            "label-1",
            "label-0",
            "label-0",
            "label-1",
            "label-1",
            "label-1",
            "label-0",
            "label-0",
            "label-1",
            "label-0",
            "label-0",
            "label-1",
            "label-1",
            "label-0",
            "label-1",
            "label-1",
            "label-1",
            "label-0",
            "label-0",
            "label-1",
            "label-1",       5
            "label-0",
            "label-0",
            "label-0",
            "label-0",
            "label-0",
            "label-1",
            "label-1",
            "label-0",
        ]
    )


def test_predict(transformer, sample_data, y_as_ids, label_encoder, expected_labels):
    X, y_labels = sample_data

    transformer.fit(X, y_as_ids)
    predictions = transformer.predict(X)

    np.testing.assert_array_equal(predictions, expected_labels)


def test_predict_proba(transformer, y_as_ids, label_encoder, sample_data):
    X, y_labels = sample_data

    transformer.fit(X, y_as_ids)
    probabilities = transformer.predict_proba(X)
    # Ensure probabilities match the expected structure

    assert probabilities == [
        [["label-0", 0.04], ["label-1", 0.96]],
        [["label-0", 0.03], ["label-1", 0.97]],
        [["label-0", 0.22], ["label-1", 0.78]],
        [["label-0", 0.99], ["label-1", 0.01]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.08], ["label-1", 0.92]],
        [["label-0", 0.07], ["label-1", 0.93]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.66], ["label-1", 0.34]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 0.99], ["label-1", 0.01]],
        [["label-0", 0.01], ["label-1", 0.99]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 0.01], ["label-1", 0.99]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.02], ["label-1", 0.98]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.02], ["label-1", 0.98]],
        [["label-0", 0.04], ["label-1", 0.96]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 0.21], ["label-1", 0.79]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.98], ["label-1", 0.02]],
        [["label-0", 1.0], ["label-1", 0.0]],
        [["label-0", 0.99], ["label-1", 0.01]],
        [["label-0", 0.0], ["label-1", 1.0]],
        [["label-0", 0.08], ["label-1", 0.92]],
        [["label-0", 1.0], ["label-1", 0.0]],
    ]
