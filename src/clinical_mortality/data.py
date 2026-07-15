"""Build an adult first-ICU-stay cohort and first-window MIMIC-IV features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CHART_FEATURES = {
    220045: "heart_rate",
    220179: "systolic_bp",
    220050: "systolic_bp",
    220180: "diastolic_bp",
    220051: "diastolic_bp",
    220181: "mean_arterial_pressure",
    220052: "mean_arterial_pressure",
    220210: "respiratory_rate",
    224690: "respiratory_rate",
    220277: "spo2",
    223762: "temperature_c",
    223761: "temperature_f",
    226755: "gcs_total",
}

LAB_FEATURES = {
    51300: "wbc",
    51301: "wbc",
    51222: "hemoglobin",
    51265: "platelets",
    50912: "creatinine",
    51006: "bun",
    50983: "sodium",
    50971: "potassium",
    50882: "bicarbonate",
    50813: "lactate",
    50931: "glucose",
    50885: "bilirubin_total",
}

AGGREGATIONS = ("min", "max", "mean", "first", "last")


def build_cohort(mimic_dir: Path, observation_hours: int = 24) -> pd.DataFrame:
    """Select each adult patient's first ICU stay and attach mortality label."""
    mimic_dir = Path(mimic_dir)
    stays = pd.read_csv(
        mimic_dir / "icu/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    admissions = pd.read_csv(
        mimic_dir / "hosp/admissions.csv.gz",
        usecols=["hadm_id", "hospital_expire_flag"],
    )
    patients = pd.read_csv(
        mimic_dir / "hosp/patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )

    cohort = stays.merge(admissions, on="hadm_id", validate="many_to_one")
    cohort = cohort.merge(patients, on="subject_id", validate="many_to_one")
    cohort = cohort.sort_values(["subject_id", "intime"]).drop_duplicates(
        "subject_id", keep="first"
    )
    cohort["age"] = (
        cohort["anchor_age"] + cohort["intime"].dt.year - cohort["anchor_year"]
    )
    cohort = cohort.loc[cohort["age"] >= 18].copy()
    cohort["window_end"] = cohort["intime"] + pd.to_timedelta(observation_hours, unit="h")
    cohort["window_end"] = cohort[["window_end", "outtime"]].min(axis=1)
    cohort = cohort.rename(columns={"hospital_expire_flag": "mortality"})
    return cohort[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "window_end",
            "age",
            "gender",
            "mortality",
        ]
    ].reset_index(drop=True)


def _read_windowed_events(
    path: Path,
    cohort: pd.DataFrame,
    item_mapping: dict[int, str],
    join_column: str,
    chunksize: int,
) -> pd.DataFrame:
    """Read only selected numeric events occurring inside the ICU window."""
    lookup_columns = [join_column, "intime", "window_end"]
    if join_column != "stay_id":
        lookup_columns.insert(1, "stay_id")
    lookup = cohort[lookup_columns].drop_duplicates(join_column)
    valid_ids = set(lookup[join_column])
    selected: list[pd.DataFrame] = []

    usecols = [join_column, "itemid", "charttime", "valuenum"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.loc[
            chunk["itemid"].isin(item_mapping)
            & chunk[join_column].isin(valid_ids)
            & chunk["valuenum"].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.merge(lookup, on=join_column, how="inner", validate="many_to_one")
        chunk = chunk.loc[
            chunk["charttime"].between(chunk["intime"], chunk["window_end"])
        ]
        if chunk.empty:
            continue
        chunk["feature"] = chunk["itemid"].map(item_mapping)
        selected.append(chunk[["stay_id", "charttime", "feature", "valuenum"]])

    if not selected:
        return pd.DataFrame(columns=["stay_id", "charttime", "feature", "valuenum"])
    return pd.concat(selected, ignore_index=True)


def _aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["stay_id"])
    events = events.sort_values(["stay_id", "feature", "charttime"])
    grouped = (
        events.groupby(["stay_id", "feature"], observed=True)["valuenum"]
        .agg(AGGREGATIONS)
        .unstack("feature")
    )
    grouped.columns = [f"{feature}_{stat}" for stat, feature in grouped.columns]
    return grouped.reset_index()


def build_feature_table(
    mimic_dir: Path,
    observation_hours: int = 24,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Create one model-ready row per patient from vitals, labs, and demographics."""
    mimic_dir = Path(mimic_dir)
    cohort = build_cohort(mimic_dir, observation_hours)

    chart = _read_windowed_events(
        mimic_dir / "icu/chartevents.csv.gz",
        cohort,
        CHART_FEATURES,
        join_column="stay_id",
        chunksize=chunksize,
    )
    # Put Fahrenheit and Celsius measurements on one clinical scale.
    fahrenheit = chart["feature"].eq("temperature_f")
    chart.loc[fahrenheit, "valuenum"] = (chart.loc[fahrenheit, "valuenum"] - 32.0) * 5 / 9
    chart.loc[fahrenheit, "feature"] = "temperature_c"

    labs = _read_windowed_events(
        mimic_dir / "hosp/labevents.csv.gz",
        cohort,
        LAB_FEATURES,
        join_column="hadm_id",
        chunksize=chunksize,
    )

    features = cohort[["subject_id", "hadm_id", "stay_id", "age", "gender", "mortality"]]
    features = features.merge(_aggregate_events(chart), on="stay_id", how="left")
    features = features.merge(_aggregate_events(labs), on="stay_id", how="left")
    return features


def assign_patient_splits(
    features: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign 60/20/10/10 train/calibration/validation/test splits."""
    train, remainder = train_test_split(
        features,
        test_size=0.40,
        stratify=features["mortality"],
        random_state=random_state,
    )
    calibration, remainder = train_test_split(
        remainder,
        test_size=0.50,
        stratify=remainder["mortality"],
        random_state=random_state,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        stratify=remainder["mortality"],
        random_state=random_state,
    )
    parts = []
    for name, frame in (
        ("train", train),
        ("calibration", calibration),
        ("validation", validation),
        ("test", test),
    ):
        part = frame.copy()
        part["split"] = name
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if result["subject_id"].duplicated().any():
        raise RuntimeError("patient leakage detected across splits")
    return result.sample(frac=1, random_state=random_state).reset_index(drop=True)
