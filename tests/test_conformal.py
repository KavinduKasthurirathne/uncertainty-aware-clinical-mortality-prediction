import numpy as np

from clinical_mortality.conformal import (
    SplitConformalClassifier,
    tune_abstention_threshold,
)
from clinical_mortality.evaluation import (
    evaluate_prediction_sets,
    expected_calibration_error,
)


def test_finite_sample_quantile_and_prediction_sets() -> None:
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.55, 0.45]]
    )
    labels = np.array([0, 0, 1, 1, 1])
    conformal = SplitConformalClassifier(alpha=0.2).fit(probabilities, labels)

    # rank = ceil((5 + 1) * .8) = 5; maximum score is 0.55.
    assert conformal.quantile_ == 0.55
    sets = conformal.predict_sets(np.array([[0.8, 0.2], [0.52, 0.48]]))
    np.testing.assert_array_equal(sets, [[True, False], [True, True]])


def test_uncertainty_penalises_ambiguous_sets() -> None:
    probabilities = np.array([[0.9, 0.1], [0.51, 0.49]])
    prediction_sets = np.array([[True, False], [True, True]])
    scores = SplitConformalClassifier.uncertainty_scores(
        probabilities, prediction_sets
    )
    assert scores[0] < scores[1]


def test_threshold_respects_minimum_retained_fraction() -> None:
    threshold, metrics = tune_abstention_threshold(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 0, 0, 1]),
        uncertainty=np.array([0.1, 0.2, 0.8, 0.3]),
        min_retained_fraction=0.75,
    )
    assert threshold == 0.3
    assert metrics["retained_fraction"] == 0.75
    assert metrics["selective_accuracy"] == 1.0


def test_coverage_and_ece() -> None:
    labels = np.array([0, 1])
    sets = np.array([[True, False], [True, True]])
    metrics = evaluate_prediction_sets(labels, sets)
    assert metrics["empirical_coverage"] == 1.0
    assert metrics["mean_set_size"] == 1.5
    assert expected_calibration_error(labels, np.array([0.0, 1.0])) == 0.0
