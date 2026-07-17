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

The pipeline combines three components:

1. **Base Predictive Model** - An XGBoost classifier trained on structured ICU patient data (vital signs, lab values, demographics) from the first 24 hours of admission.
2. **Conformal Uncertainty Estimation** - Split Conformal Prediction is applied as a post-hoc wrapper to generate prediction sets with statistically guaranteed marginal coverage (90% at α = 0.1).
3. **Selective Prediction / Abstention** - A threshold-based mechanism defers high-uncertainty predictions to clinician review, improving reliability on the cases the model does act upon.

---

## Proposal Completion Status

| Proposal item | Status |
|---------------|--------|
| MIMIC-IV cohort + first-24h features | Done |
| XGBoost primary model | Done |
| Random Forest baseline | Done |
| Split Conformal Prediction | Done |
| Selective abstention | Done |
| Table 3.1 evaluation metrics | Done |
| Three-way comparison (XGBoost / +Conformal / +Selective) | Done |
| eICU external validation | **Excluded from current scope** |
| Optional PhysioNet benchmarks | Not used |

The current project scope is **MIMIC-IV internal training, conformal calibration, selective prediction, and evaluation**. Cross-institutional eICU validation is intentionally not included in this version.

---

## Dataset

| Dataset | Role in this project |
|---------|----------------------|
| **MIMIC-IV 3.1** | Primary training, calibration, validation, and test |
| eICU | Not used in the current run |

Raw MIMIC-IV layout expected:

```text
mimic-iv-3.1/
  icu/   (icustays, chartevents, ...)
  hosp/  (admissions, patients, labevents, ...)
```

Raw MIMIC files and generated patient-level outputs are ignored by Git and must never be committed or uploaded.

---

## Methodology

### Data flow

```text
MIMIC-IV raw tables
        |
        v
Adult first ICU stay + in-hospital mortality label
        |
        v
First-24h vitals + labs + demographics
        |
        v
Feature aggregates (min / max / mean / first / last)
        |
        v
Patient-level stratified splits
  train 60% | calibration 20% | validation 10% | test 10%
        |
        +--> XGBoost / Random Forest baselines
        |
        +--> Split Conformal Prediction (alpha = 0.1)
        |
        +--> Validation-tuned abstention threshold
        |
        v
Test metrics + patient-level predictions
```

### Cohort construction

- Keep each adult patient's **first ICU stay** (`age >= 18`) to avoid patient leakage.
- Label: `hospital_expire_flag` from MIMIC-IV admissions (in-hospital mortality).
- Observation window: first **24 hours** after ICU `intime` (capped by `outtime`).

### Features

Demographics: `age`, `gender`

Vitals (from `chartevents`): heart rate, systolic/diastolic/mean arterial pressure, respiratory rate, SpO₂, temperature (°C), GCS total

Labs (from `labevents`): WBC, hemoglobin, platelets, creatinine, BUN, sodium, potassium, bicarbonate, lactate, glucose, total bilirubin

For each vital/lab, compute: **min, max, mean, first, last**.

### Models and uncertainty

1. **Baselines**
   - XGBoost (primary)
   - Random Forest (secondary)
2. **Split Conformal Prediction**
   - Nonconformity score: `1 - P(true class)`
   - Finite-sample quantile correction at α = 0.1
   - Output: prediction sets with target 90% marginal coverage
   - Uncertainty score combines probability margin and non-singleton set size
3. **Selective abstention**
   - Threshold `τ` tuned on the validation set
   - Objective: maximise selective accuracy while retaining at least 70% of cases (`--min-retained 0.7`)
   - Decision rule:
     - `uncertainty(x) ≤ τ` → issue Mortality/Survival prediction
     - `uncertainty(x) > τ` → abstain → clinician review

### Mathematical formulation

For a patient \(x\), XGBoost produces class probabilities
\(\hat{p}(y \mid x)\) for survival (\(y=0\)) and mortality (\(y=1\)). The
point prediction is:

\[
\hat{y}(x) = \arg\max_y \hat{p}(y \mid x)
\]

For each calibration patient \((x_i, y_i)\), the conformal nonconformity score
is:

\[
s_i = 1 - \hat{p}(y_i \mid x_i)
\]

The calibration quantile uses the finite-sample corrected rank
\(\lceil(n+1)(1-\alpha)\rceil\). A class is included in the prediction set when:

\[
1 - \hat{p}(y \mid x) \leq \hat{q}
\]

The implementation assigns a continuous uncertainty score:

\[
u(x) = 1 - \text{probability margin}(x)
       + \mathbb{1}\{|\Gamma(x)| \neq 1\}
\]

Here, \(\Gamma(x)\) is the conformal prediction set. A small probability margin
or a non-singleton set increases uncertainty. The final selective decision is:

\[
u(x) \leq \tau \Rightarrow \text{predict}, \qquad
u(x) > \tau \Rightarrow \text{abstain}
\]

### Evaluation metrics (Table 3.1)

| Metric | Purpose |
|--------|---------|
| AUROC | Discriminative performance |
| Accuracy | Overall classification accuracy |
| ECE | Calibration quality of probabilities |
| Empirical Coverage | Fraction of true labels in conformal sets |
| Mean Prediction Set Size | Sharpness of conformal sets |
| Abstention Rate | Fraction deferred to clinician review |
| Selective Accuracy | Accuracy on non-abstained predictions |

Class-conditional coverage (survival vs mortality) is also reported because marginal coverage can hide per-class imbalance.

---

## Installation

Python 3.11 or newer is recommended.

```bash
cd uncertainty-aware-clinical-mortality-prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,plots]"
```

Verify the CLI:

```bash
clinical-mortality --help
```

---

## Steps Followed (Training Documentation)

### How the project was carried out

1. The MIMIC-IV schema and relevant item IDs were identified.
2. ICU stays, admissions, and patient records were joined to create the cohort.
3. The first adult ICU stay for each patient was retained.
4. Numeric vital-sign and laboratory events inside the first 24 hours were
   extracted from the large compressed event tables in chunks.
5. Repeated measurements were converted into fixed-length tabular features
   using min/max/mean/first/last aggregation.
6. Patients were divided into four disjoint, mortality-stratified subsets.
7. Numeric missing values were median-imputed and gender was one-hot encoded.
8. XGBoost and Random Forest baselines were trained and evaluated.
9. XGBoost probabilities on the calibration split were used to fit the
   conformal threshold.
10. The validation split was used to select an abstention threshold while
    retaining at least 70% of patients.
11. All final metrics were calculated once on the untouched test split.
12. The trained components and patient-level test decisions were saved for
    reproducibility and analysis.

### Step 1 — Unit tests (sanity check)

```bash
pytest
# or: .venv/bin/pytest -q
```

These tests cover conformal quantile / prediction-set logic, uncertainty scoring, abstention-threshold selection, and a synthetic first-24h feature-window check. They do **not** require the full MIMIC tables.

### Step 2 — Build the MIMIC-IV feature table

```bash
clinical-mortality build-features \
  --mimic-dir mimic-iv-3.1 \
  --output data/processed/mimic_features.parquet \
  --hours 24
```

This scans large event tables (`chartevents`, `labevents`) in chunks and can take a long time (often 1–3+ hours). Progress is silent until completion.

Expected console summary:

```text
Wrote 65,366 patient rows and 102 columns to data/processed/mimic_features.parquet
             count      mean
split
calibration  13073  0.108391
test          6537  0.108307
train        39219  0.108417
validation    6537  0.108460
```

### Step 3 — Train baselines, conformal wrapper, and selective model

```bash
clinical-mortality train \
  --features data/processed/mimic_features.parquet \
  --output-dir outputs \
  --alpha 0.1 \
  --min-retained 0.7
```

What this command does:

1. Fits preprocessing (median imputation for numeric features; one-hot encoding for `gender`)
2. Trains XGBoost and Random Forest on the train split
3. Calibrates Split Conformal Prediction on the calibration split
4. Tunes the abstention threshold on the validation split
5. Evaluates all variants on the held-out test split
6. Writes metrics, predictions, and a saved experiment artifact

### Step 4 — Inspect results

```bash
cat outputs/metrics.json
head outputs/test_predictions.csv
```

---

## Generated Artifacts

| Path | Description |
|------|-------------|
| `data/processed/mimic_features.parquet` | Patient-level feature matrix with split labels (65,366 × 102) |
| `outputs/metrics.json` | Baseline, conformal, and selective metrics |
| `outputs/test_predictions.csv` | Per-patient probabilities, prediction sets, uncertainty, abstention flags |
| `outputs/experiment.joblib` | Fitted preprocessor, models, conformal calibrator, and abstention threshold |

### Columns in `test_predictions.csv`

| Column | Meaning |
|--------|---------|
| `subject_id`, `hadm_id`, `stay_id` | Patient / admission / ICU stay identifiers |
| `mortality` | True label (0 = survival, 1 = death) |
| `mortality_probability` | XGBoost predicted P(death) |
| `point_prediction` | Argmax class prediction |
| `set_survival`, `set_mortality` | Whether each class is in the conformal set |
| `uncertainty_score` | Scalar uncertainty used for abstention |
| `abstained` | `True` if deferred to clinician review |

---

## Results (MIMIC-IV held-out test set)

- Cohort size: **65,366** adult first ICU stays
- Test size: **6,537** patients
- Mortality prevalence: ~10.8% across splits
- Significance level: **α = 0.1** (90% target coverage)
- Abstention threshold: **τ ≈ 0.555** (tuned on validation)

| Variant | AUROC | Accuracy | ECE | Empirical Coverage | Mean Set Size | Abstention Rate | Selective Accuracy |
|---------|-------|----------|-----|--------------------|---------------|-----------------|--------------------|
| XGBoost | 0.8940 | 0.8411 | 0.1789 | — | — | — | — |
| Random Forest | 0.8821 | 0.9081 | 0.1069 | — | — | — | — |
| XGBoost + Conformal | 0.8940 | 0.8411 | 0.1789 | **0.9019** | 1.1351 | — | — |
| XGBoost + Conformal + Selective | 0.8940 | 0.8411 | 0.1789 | 0.9019 | 1.1351 | **0.2902** | **0.9278** |

Additional selective metrics:

- Retained fraction: **0.7098**
- Selective AUROC: **0.9460**
- Validation selective accuracy (used for threshold tuning): **0.9257**

Class-conditional conformal coverage:

| Class | Coverage |
|-------|----------|
| Survival (class 0) | 0.9113 |
| Mortality (class 1) | **0.8249** |

### Interpretation

- Marginal conformal coverage (**90.19%**) meets the nominal 90% target.
- Selective abstention improves accuracy on retained cases from **0.841** (XGBoost point predictions) to **0.928**, at the cost of deferring about **29%** of test patients.
- Mortality-class coverage (**82.49%**) is below the 90% target even though overall coverage is valid. Marginal guarantees can hide clinically important per-class imbalance; this is reported explicitly and remains a limitation of standard split conformal prediction.

### Achieved prediction analysis

The held-out test set contains 5,829 survivors and 708 deaths. The following
analysis was calculated from `outputs/test_predictions.csv`.

#### XGBoost point predictions (all test patients)

| True outcome | Predicted survival | Predicted mortality |
|--------------|--------------------|---------------------|
| Survival | 4,973 | 856 |
| Mortality | 183 | 525 |

- Mortality sensitivity: **74.15%** (525/708 deaths detected)
- Survival specificity: **85.32%** (4,973/5,829 survivors identified)
- Mean predicted mortality probability:
  - True survivors: **0.238**
  - True deaths: **0.690**

The model separates the classes well (AUROC 0.894), but its class-weighted
training produces more positive predictions and therefore lower raw accuracy
than the Random Forest. Accuracy alone is not sufficient for this imbalanced
clinical task; AUROC, calibration, sensitivity, and uncertainty must be
considered together.

#### Conformal prediction sets

- **86.49%** of patients received a singleton set containing one class.
- **13.51%** received the ambiguous set `{Survival, Mortality}`.
- No empty sets were produced in this run.
- Overall coverage reached **90.19%**, but mortality coverage was lower
  (**82.49%**) than survival coverage (**91.13%**).

The prediction set does not replace the mortality probability. It answers a
different question: which outcome labels remain plausible under the calibrated
coverage rule?

#### Selective predictions (non-abstained patients)

| True outcome | Predicted survival | Predicted mortality |
|--------------|--------------------|---------------------|
| Survival | 3,922 | 269 |
| Mortality | 66 | 383 |

- Retained patients: **4,640 / 6,537 (70.98%)**
- Selective mortality sensitivity: **85.30%**
- Selective survival specificity: **93.58%**
- Selective accuracy: **92.78%**
- Abstention among true survivors: **28.10%**
- Abstention among true deaths: **36.58%**

These results show that uncertain cases contain a larger share of difficult
errors. Deferring them improves performance on the retained subset. However,
the higher abstention rate among patients who died must be considered when
designing a clinical workflow: abstention means escalation to human review,
not denial of care or removal from evaluation.

### How to interpret one prediction row

Example (identifiers omitted):

```text
mortality=0
mortality_probability=0.2858
point_prediction=0
set_survival=True
set_mortality=False
uncertainty_score=0.5717
abstained=True
```

Interpretation:

1. The patient survived (`mortality=0`).
2. XGBoost assigned a 28.58% mortality probability and selected survival as
   the point prediction.
3. The conformal set contained only survival.
4. Despite the correct singleton set, the probability margin was not large
   enough to pass the stricter selective threshold (\(\tau \approx 0.555\)).
5. The system therefore abstained and would request clinician review.

This illustrates why the point prediction, conformal set, and abstention
decision should be read together. The current project explains *prediction
confidence and set membership*; it does not yet explain which individual
clinical features caused a prediction. Per-feature explanations such as SHAP
are listed as future work.

### Main conclusion

The experiment demonstrates the intended project concept on MIMIC-IV:
conformal prediction provides a measurable coverage target, and selective
abstention improves reliability on predictions retained for automated use.
The system should still be treated as a research prototype because calibration
quality, mortality-class coverage, external validity, and clinical utility
require further study.

---

## Repository Layout

```text
src/clinical_mortality/
  cli.py          # build-features and train commands
  data.py         # MIMIC-IV cohort + feature extraction + splits
  modeling.py     # preprocessing, XGBoost/RF, experiment orchestration
  conformal.py    # split conformal + abstention threshold tuning
  evaluation.py   # AUROC, ECE, coverage, selective metrics
tests/            # unit tests for conformal logic and feature windowing
data/processed/   # generated feature parquet (gitignored)
outputs/          # generated metrics / predictions / model artifact (gitignored)
mimic-iv-3.1/     # raw MIMIC-IV data (gitignored)
```

---

## Limitations and Scope Notes

1. **No eICU external validation** in the current project run (excluded by design for this documentation stage).
2. **Marginal coverage only** is formally guaranteed; mortality-class coverage is lower than the nominal 90%.
3. **Retrospective single-database evaluation** on MIMIC-IV; results may not generalise to other hospitals or care processes without external validation.
4. **Research prototype only** — not intended for clinical deployment or patient care decisions.
5. Feature extraction depends on selected MIMIC item IDs; different item mappings can change feature availability and model performance.

---

## Future Work

The recommended next steps, in priority order, are:

1. **Improve mortality-class coverage**
   - Evaluate class-conditional (Mondrian) conformal prediction.
   - Compare mortality and survival coverage at several α values.
2. **Improve probability calibration**
   - Compare isotonic regression and Platt scaling using validation-only data.
   - Recalculate ECE, Brier score, and reliability diagrams.
3. **Analyse the selective-risk trade-off**
   - Plot retained coverage versus selective accuracy, sensitivity, and
     abstention rate across many thresholds.
   - Select thresholds using an explicit clinical cost or safety requirement.
4. **Add prediction explanations**
   - Generate global XGBoost feature importance.
   - Use SHAP values for per-patient explanations of high mortality risk.
   - Keep explanation confidence separate from conformal uncertainty.
5. **Perform robustness and subgroup analysis**
   - Report performance by sex, age group, mortality class, missingness, and
     ICU type.
   - Test temporal splits and simulated missing-data or distribution shifts.
6. **External validation (optional future extension)**
   - Harmonise MIMIC-IV features with another hospital dataset such as eICU.
   - Reassess probability calibration and conformal coverage after the shift.
7. **Clinical workflow evaluation**
   - Define how abstained cases are prioritised for review.
   - Conduct prospective evaluation, usability testing, and governance review
     before any clinical deployment.

---

## Reproducing the Experiment End-to-End

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,plots]"

# 2. Tests
pytest

# 3. Features (slow)
clinical-mortality build-features \
  --mimic-dir mimic-iv-3.1 \
  --output data/processed/mimic_features.parquet \
  --hours 24

# 4. Train + evaluate
clinical-mortality train \
  --features data/processed/mimic_features.parquet \
  --output-dir outputs \
  --alpha 0.1 \
  --min-retained 0.7

# 5. Review
cat outputs/metrics.json
```

---

## Department of Computer Science and Engineering

University of Moratuwa, Sri Lanka - May 2026
