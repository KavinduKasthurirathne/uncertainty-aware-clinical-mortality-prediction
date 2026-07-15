"""Finite-sample split conformal classification and selective prediction."""

from __future__ import annotations

import numpy as np


class SplitConformalClassifier:
    """Split conformal wrapper using 1 - P(true class) nonconformity."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.quantile_: float | None = None
        self.n_classes_: int | None = None

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray) -> "SplitConformalClassifier":
        probabilities = self._validate_probabilities(probabilities)
        y_true = np.asarray(y_true, dtype=int)
        if len(y_true) != len(probabilities):
            raise ValueError("probabilities and y_true must have equal length")
        if np.any((y_true < 0) | (y_true >= probabilities.shape[1])):
            raise ValueError("y_true contains an invalid class index")

        scores = 1.0 - probabilities[np.arange(len(y_true)), y_true]
        # Finite-sample corrected order statistic:
        # ceil((n + 1) * (1 - alpha)), capped at n.
        rank = min(int(np.ceil((len(scores) + 1) * (1.0 - self.alpha))), len(scores))
        self.quantile_ = float(np.sort(scores)[rank - 1])
        self.n_classes_ = probabilities.shape[1]
        return self

    def predict_sets(self, probabilities: np.ndarray) -> np.ndarray:
        """Return a boolean matrix: row i contains all labels in Γ(x_i)."""
        if self.quantile_ is None or self.n_classes_ is None:
            raise RuntimeError("fit must be called before predict_sets")
        probabilities = self._validate_probabilities(probabilities)
        if probabilities.shape[1] != self.n_classes_:
            raise ValueError("number of classes differs from calibration data")
        return (1.0 - probabilities) <= self.quantile_

    @staticmethod
    def uncertainty_scores(probabilities: np.ndarray, prediction_sets: np.ndarray) -> np.ndarray:
        """Continuous score combining ambiguity and conformal set non-singletons.

        Scores range from 0 (large probability margin and singleton set) to 2
        (small margin and empty/multi-label set).
        """
        probabilities = SplitConformalClassifier._validate_probabilities(probabilities)
        prediction_sets = np.asarray(prediction_sets, dtype=bool)
        if prediction_sets.shape != probabilities.shape:
            raise ValueError("prediction_sets must match probabilities shape")
        sorted_probabilities = np.sort(probabilities, axis=1)
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        non_singleton = prediction_sets.sum(axis=1) != 1
        return (1.0 - margin) + non_singleton.astype(float)

    @staticmethod
    def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("probabilities must have shape (n_samples, n_classes)")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must be finite")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("probabilities must be between 0 and 1")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each probability row must sum to 1")
        return probabilities


def tune_abstention_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    min_retained_fraction: float = 0.7,
) -> tuple[float, dict[str, float]]:
    """Maximise validation accuracy while retaining a minimum case fraction."""
    if not 0 < min_retained_fraction <= 1:
        raise ValueError("min_retained_fraction must be in (0, 1]")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if not (len(y_true) == len(y_pred) == len(uncertainty)):
        raise ValueError("all arrays must have equal length")

    candidates = np.unique(uncertainty)
    best: tuple[tuple[float, float], float, dict[str, float]] | None = None
    for threshold in candidates:
        retained = uncertainty <= threshold
        retained_fraction = float(retained.mean())
        if retained_fraction < min_retained_fraction or not retained.any():
            continue
        accuracy = float((y_true[retained] == y_pred[retained]).mean())
        metrics = {
            "threshold": float(threshold),
            "retained_fraction": retained_fraction,
            "abstention_rate": 1.0 - retained_fraction,
            "selective_accuracy": accuracy,
        }
        key = (accuracy, retained_fraction)
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)

    if best is None:
        raise ValueError("no threshold satisfies min_retained_fraction")
    return best[1], best[2]
