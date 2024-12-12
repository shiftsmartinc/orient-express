from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    A wrapper class that integrates a model and a label encoder to handle
    encoded predictions and their corresponding probabilities.

    Attributes:
        model (object): A machine learning model with `fit`, `predict`, and `predict_proba` methods.
        label_encoder (object): A label encoder that maps encoded class labels to their original string labels.

    Methods:
        fit(X, y):
            Trains the model on the provided features and labels.
        predict(X):
            Predicts the classes for the given features and returns the original string labels.
        predict_proba(X):
            Returns the probabilities for each class for the given features, along with their original class labels.
    """

    def __init__(self, model: BaseEstimator, label_encoder: LabelEncoder):
        """
        Initializes the LabelEncoderTransformer.

        Args:
            model (BaseEstimator): A machine learning model.
            label_encoder (LabelEncoder): A label encoder with `fit` and `inverse_transform` methods.
        """
        self.model = model
        self.label_encoder = label_encoder

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            self: Fitted LabelEncoderTransformer instance.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the target labels and returns the original string labels.

        Args:
            X (array-like): Feature matrix for predictions.

        Returns:
            array-like: Predicted class labels in their original form.
        """
        encoded_predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(encoded_predictions)

    def predict_proba(self, X):
        """
        Returns class probabilities along with their original labels.

        Args:
            X (array-like): Feature matrix for predictions.

        Returns:
            list: A list of lists, where each inner list contains `[class_name, probability]` pairs.
        """
        # Get raw probabilities
        probabilities = self.model.predict_proba(X)

        # Get all class labels
        class_names = self.label_encoder.classes_
        # Combine class names with probabilities
        combined_output = [
            [[class_name, prob] for class_name, prob in zip(class_names, sample_probs)]
            for sample_probs in probabilities
        ]
        return combined_output
