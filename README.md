# Uncertainty-Aware Clinical Mortality Prediction using Conformal Prediction and Selective Prediction

**CS5801 Advanced AI - Group Project | University of Moratuwa**

---

## Group

**CS5801_G30**

| Name | Index |
|------|-------|
| K.K.I. Kasthurirathne | 268465B |
| T.M.L.A.S. Thennakoon | 268503R |
| W.K.A. Pehesarani | 268475F |

---

## Project Description

This project develops an uncertainty-aware clinical mortality prediction system for ICU settings. Standard machine learning classifiers (e.g., XGBoost, Random Forest) produce single-point mortality risk estimates without any indication of predictive confidence - a known safety concern in critical care environments.

The proposed pipeline addresses this by combining three components:

1. **Base Predictive Model** - An XGBoost classifier trained on structured ICU patient data (vital signs, lab values, demographics) from the first 24-48 hours of admission.
2. **Conformal Uncertainty Estimation** - Split Conformal Prediction is applied as a post-hoc wrapper to generate prediction sets with statistically guaranteed marginal coverage (e.g., 90% at α = 0.1).
3. **Selective Prediction / Abstention** - A threshold-based mechanism defers high-uncertainty predictions to clinician review, improving reliability on the cases the model does act upon.

---

## Datasets

| Dataset | Role |
|---------|------|
| MIMIC-III / MIMIC-IV | Primary training & validation |
| eICU Collaborative Research Database | External validation |

All datasets are accessible via [PhysioNet](https://physionet.org/).

---

## Evaluation Metrics

- AUROC, Accuracy
- Expected Calibration Error (ECE)
- Empirical Coverage, Mean Prediction Set Size
- Rejection / Abstention Rate, Selective Accuracy

---

## Implemented Workflow

The code uses each adult patient's first ICU stay to avoid patient leakage. It
extracts demographics, vital signs, and laboratory measurements from the first
24 hours, then assigns stratified 60/20/10/10 train/calibration/validation/test
splits.

The experiment compares:

1. XGBoost and Random Forest probability baselines.
2. XGBoost with finite-sample Split Conformal Prediction.
3. XGBoost with conformal sets and validation-tuned selective abstention.

Class-conditional conformal coverage is reported because marginal coverage can
hide clinically important mortality/survival imbalances.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,plots]"
```

## Run the Pipeline

The expected raw-data layout is the standard MIMIC-IV 3.1 structure:
`mimic-iv-3.1/icu/*.csv.gz` and `mimic-iv-3.1/hosp/*.csv.gz`.

Build the first-24-hour patient feature table:

```bash
clinical-mortality build-features \
  --mimic-dir mimic-iv-3.1 \
  --output data/processed/mimic_features.parquet \
  --hours 24
```

This scans the large event tables in chunks and can take substantial time.
Train and evaluate all model variants:

```bash
clinical-mortality train \
  --features data/processed/mimic_features.parquet \
  --output-dir outputs \
  --alpha 0.1 \
  --min-retained 0.7
```

The command writes:

- `outputs/metrics.json`: baseline, conformal, and selective metrics.
- `outputs/test_predictions.csv`: patient-level probabilities, sets, uncertainty,
  and abstention decisions.
- `outputs/experiment.joblib`: fitted preprocessing, models, conformal calibrator,
  and abstention threshold.

Run tests with:

```bash
pytest
```

Raw MIMIC files and generated patient-level outputs are ignored by Git and must
never be committed or uploaded.

---

## Department of Computer Science and Engineering
University of Moratuwa, Sri Lanka - May 2026
