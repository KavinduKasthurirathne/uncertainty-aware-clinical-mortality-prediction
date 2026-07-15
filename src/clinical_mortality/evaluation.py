"""Evaluation metrics for probability, conformal, and selective predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray, positive_probability: np.ndarray, n_bins: int = 10
) -> float:
    """Compute equal-width Expected Calibration Error for binary outcomes."""
    y_true = np.asarray(y_true, dtype=int)
    positive_probability = np.asarray(positive_probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Include probability 1.0 in the final bin.
    bins = np.minimum(np.digitize(positive_probability, edges[1:-1]), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bins == bin_index
        if mask.any():
            ece += mask.mean() * abs(
                positive_probability[mask].mean() - y_true[mask].mean()
            )
    return float(ece)


def evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities[:, 1])),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "ece": expected_calibration_error(y_true, probabilities[:, 1]),
    }


def evaluate_prediction_sets(
    y_true: np.ndarray, prediction_sets: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    covered = prediction_sets[np.arange(len(y_true)), y_true]
    metrics = {
        "empirical_coverage": float(covered.mean()),
        "mean_set_size": float(prediction_sets.sum(axis=1).mean()),
    }
    for class_index in np.unique(y_true):
        class_mask = y_true == class_index
        metrics[f"class_{class_index}_coverage"] = float(covered[class_mask].mean())
    return metrics


def evaluate_selective(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    retained = np.asarray(uncertainty) <= threshold
    predictions = np.argmax(probabilities, axis=1)
    metrics = {
        "retained_fraction": float(retained.mean()),
        "abstention_rate": float(1.0 - retained.mean()),
        "selective_accuracy": float(accuracy_score(y_true[retained], predictions[retained])),
    }
    if len(np.unique(np.asarray(y_true)[retained])) == 2:
        metrics["selective_auroc"] = float(
            roc_auc_score(np.asarray(y_true)[retained], probabilities[retained, 1])
        )
    else:
        metrics["selective_auroc"] = float("nan")
    return metrics
