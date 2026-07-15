from pathlib import Path

import pandas as pd

from clinical_mortality.data import build_feature_table


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")


def test_build_feature_table_uses_only_first_24_hours(tmp_path: Path) -> None:
    _write(
        pd.DataFrame(
            {
                "subject_id": [1],
                "hadm_id": [10],
                "stay_id": [100],
                "intime": ["2020-01-01 00:00:00"],
                "outtime": ["2020-01-03 00:00:00"],
                "los": [2.0],
                "first_careunit": ["MICU"],
                "last_careunit": ["MICU"],
            }
        ),
        tmp_path / "icu/icustays.csv.gz",
    )
    _write(
        pd.DataFrame({"hadm_id": [10], "subject_id": [1], "hospital_expire_flag": [1]}),
        tmp_path / "hosp/admissions.csv.gz",
    )
    _write(
        pd.DataFrame(
            {
                "subject_id": [1],
                "gender": ["F"],
                "anchor_age": [50],
                "anchor_year": [2020],
            }
        ),
        tmp_path / "hosp/patients.csv.gz",
    )
    _write(
        pd.DataFrame(
            {
                "stay_id": [100, 100],
                "itemid": [220045, 220045],
                "charttime": ["2020-01-01 01:00:00", "2020-01-02 02:00:00"],
                "valuenum": [80.0, 200.0],
            }
        ),
        tmp_path / "icu/chartevents.csv.gz",
    )
    _write(
        pd.DataFrame(
            {
                "hadm_id": [10],
                "itemid": [50912],
                "charttime": ["2020-01-01 02:00:00"],
                "valuenum": [1.2],
            }
        ),
        tmp_path / "hosp/labevents.csv.gz",
    )

    result = build_feature_table(tmp_path, observation_hours=24, chunksize=1)

    assert len(result) == 1
    assert result.loc[0, "mortality"] == 1
    assert result.loc[0, "heart_rate_max"] == 80.0
    assert result.loc[0, "creatinine_mean"] == 1.2
