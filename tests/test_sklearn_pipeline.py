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
        n_samples=100,
        n_features=5,
        n_clusters_per_class=1,
        n_classes=3,
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


def test_predict(transformer, sample_data, y_as_ids, label_encoder):
    X, y_labels = sample_data

    transformer.fit(X, y_as_ids)
    predictions = transformer.predict(X)

    # Ensure predictions are in original class labels
    assert all(label in label_encoder.classes_ for label in predictions)


def test_predict_proba(transformer, y_as_ids, label_encoder, sample_data):
    X, y_labels = sample_data

    transformer.fit(X, y_as_ids)
    probabilities = transformer.predict_proba(X)
    # Ensure probabilities match the expected structure
    for sample_probs in probabilities:
        assert len(sample_probs) == len(label_encoder.classes_)
        for class_prob in sample_probs:
            assert isinstance(class_prob[0], str)  # Class name
            assert 0 <= class_prob[1] <= 1  # Probability is valid
