"""Tests for sentiment-model evaluation."""

import numpy as np

from sentiment_analysis.libs.evaluation import evaluate_model


class StaticClassifier:
    """Return deterministic predictions in a fixed class order."""

    classes_ = np.array(["negative", "neutral", "positive"])

    def predict(self, _features):
        """Return one prediction for each fixture row."""
        return np.array(["negative", "positive", "neutral"])


def test_evaluate_model_uses_classifier_order_for_confusion_matrix() -> None:
    """Keep confusion-matrix axes aligned with classifier class order."""
    accuracy, _, matrix, class_names = evaluate_model(
        StaticClassifier(),
        np.zeros((3, 1)),
        np.array(["neutral", "negative", "positive"]),
    )

    assert accuracy == 0
    assert class_names.tolist() == ["negative", "neutral", "positive"]
    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        ),
    )
