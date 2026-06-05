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

## Department of Computer Science and Engineering
University of Moratuwa, Sri Lanka - May 2026
