"""Train and evaluate mortality models with conformal selective prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .conformal import SplitConformalClassifier, tune_abstention_threshold
from .evaluation import (
    evaluate_prediction_sets,
    evaluate_probabilities,
    evaluate_selective,
)


NON_FEATURE_COLUMNS = {"subject_id", "hadm_id", "stay_id", "mortality", "split"}


def _make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    feature_columns = [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]
    categorical = [column for column in feature_columns if frame[column].dtype == "object"]
    numeric = [column for column in feature_columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _make_models(train_labels: pd.Series, random_state: int) -> dict[str, object]:
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    scale_pos_weight = negatives / max(positives, 1)
    return {
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def run_experiment(
    feature_path: Path,
    output_dir: Path,
    alpha: float = 0.1,
    min_retained_fraction: float = 0.7,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Run both baselines and the full conformal-abstention experiment."""
    feature_path = Path(feature_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(feature_path)
    required_splits = {"train", "calibration", "validation", "test"}
    if set(data["split"]) != required_splits:
        raise ValueError(f"split column must contain exactly {sorted(required_splits)}")

    partitions = {
        name: data.loc[data["split"] == name].reset_index(drop=True)
        for name in required_splits
    }
    feature_columns = [
        column for column in data.columns if column not in NON_FEATURE_COLUMNS
    ]
    preprocessor = _make_preprocessor(partitions["train"])
    train_x = preprocessor.fit_transform(partitions["train"][feature_columns])
    transformed = {
        name: preprocessor.transform(frame[feature_columns])
        for name, frame in partitions.items()
        if name != "train"
    }
    train_y = partitions["train"]["mortality"].to_numpy(dtype=int)
    labels = {
        name: frame["mortality"].to_numpy(dtype=int)
        for name, frame in partitions.items()
    }

    metrics: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, object] = {}
    for name, model in _make_models(partitions["train"]["mortality"], random_state).items():
        model.fit(train_x, train_y)
        fitted_models[name] = model
        test_probabilities = model.predict_proba(transformed["test"])
        metrics[name] = evaluate_probabilities(labels["test"], test_probabilities)

    model = fitted_models["xgboost"]
    calibration_probabilities = model.predict_proba(transformed["calibration"])
    validation_probabilities = model.predict_proba(transformed["validation"])
    test_probabilities = model.predict_proba(transformed["test"])

    conformal = SplitConformalClassifier(alpha=alpha).fit(
        calibration_probabilities, labels["calibration"]
    )
    validation_sets = conformal.predict_sets(validation_probabilities)
    test_sets = conformal.predict_sets(test_probabilities)
    validation_uncertainty = conformal.uncertainty_scores(
        validation_probabilities, validation_sets
    )
    test_uncertainty = conformal.uncertainty_scores(test_probabilities, test_sets)
    threshold, validation_selection = tune_abstention_threshold(
        labels["validation"],
        np.argmax(validation_probabilities, axis=1),
        validation_uncertainty,
        min_retained_fraction=min_retained_fraction,
    )

    metrics["xgboost_conformal"] = {
        **metrics["xgboost"],
        **evaluate_prediction_sets(labels["test"], test_sets),
        "alpha": alpha,
        "conformal_quantile": float(conformal.quantile_),
    }
    metrics["xgboost_conformal_selective"] = {
        **metrics["xgboost_conformal"],
        **evaluate_selective(
            labels["test"], test_probabilities, test_uncertainty, threshold
        ),
        "abstention_threshold": threshold,
        "validation_selective_accuracy": validation_selection["selective_accuracy"],
    }

    predictions = partitions["test"][["subject_id", "hadm_id", "stay_id", "mortality"]].copy()
    predictions["mortality_probability"] = test_probabilities[:, 1]
    predictions["point_prediction"] = np.argmax(test_probabilities, axis=1)
    predictions["set_survival"] = test_sets[:, 0]
    predictions["set_mortality"] = test_sets[:, 1]
    predictions["uncertainty_score"] = test_uncertainty
    predictions["abstained"] = test_uncertainty > threshold
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "models": fitted_models,
            "conformal": conformal,
            "abstention_threshold": threshold,
            "feature_columns": feature_columns,
        },
        output_dir / "experiment.joblib",
    )
    return metrics
