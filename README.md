# AI-Based Clinical Decision Support System for Genomic Variant Classification (TEKNOFEST 2026)

**Team:** Biyoinformatikçiler  
**Competition:** TEKNOFEST 2026 — Health Artificial Intelligence, University and Above  
**Domain:** Bioinformatics · Clinical machine learning · Genomic variant classification

## Overview

This project investigates machine-learning methods for classifying genomic variants as **Pathogenic (1)** or **Benign (0)** across four competition panels: **MASTER**, **CANCER** (hereditary cancer), **PAH**, and **CFTR**. It was developed for the TEKNOFEST 2026 Health AI competition, which deliberately introduced a substantial shift in class prevalence between the labeled training sets and the hidden-label final test sets. The system combines panel-specific preprocessing, feature selection, ensemble classifiers, probability calibration where applicable, and prevalence-aware decision thresholds.

**Scope:** Research and competition prototype. Model predictions are not independently validated clinical diagnoses and should not be used for patient care.

## Dataset and access

The training data were supplied by the TEKNOFEST 2026 competition organizers as four labeled CSV files. Each row describes a variant using anonymized feature names rather than its chromosome or genomic position. The supplied feature groups were:

| Prefix | Feature family |
|---|---|
| `AL_` | Allele frequencies and population-related features |
| `AA_` | Amino-acid substitution information |
| `EK_` | Evolutionary conservation-related features |
| `CAT_` | Categorical metadata |

`Variant_ID` identifies a row; `Label` is its training label (`1` = Pathogenic, `0` = Benign). `Variant_ID` is **not** used as a model feature.

### Labeled training-set composition

| Panel | Pathogenic | Benign | Total |
|:--|--:|--:|--:|
| MASTER | 2,149 | 782 | 2,931 |
| CANCER | 268 | 120 | 388 |
| PAH | 310 | 62 | 372 |
| CFTR | 90 | 21 | 111 |
| **Total** | **2,817** | **985** | **3,802** |

The organizer described the hidden-label test sets as **benign-majority**, in contrast to the pathogenic-majority training sets. The approximate *announced* test distributions were MASTER 500/3,000, CANCER 100/500, PAH 100/250, and CFTR 20/100 (Pathogenic/Benign). These are disclosed competition design figures, **not observed ground-truth labels from the final test data**.

**Data availability:** The organizer-provided training and final test CSV files are **not included in this repository**. This repository documents the dataset structure and training methodology without redistributing those files. The notebook expects the four original training CSV files in the user's own Google Drive under `MyDrive/Colab Notebooks/`, with these filenames:

```text
YARISMA_TRAIN_MASTER.csv
YARISMA_TRAIN_KANSER.csv
YARISMA_TRAIN_PAH.csv
YARISMA_TRAIN_CFTR.csv
```

Access to the original data is subject to the organizer's distribution rules. No final competition test records should be committed to this repository.

## Methodology

The final training workflow is implemented in [`Mutation_Prediction.ipynb`](Mutation_Prediction.ipynb). It uses **five-fold stratified out-of-fold (OOF) validation** to generate predictions used for threshold-policy selection and evaluation. During OOF evaluation, preprocessing and feature selection are fitted on the training portion of each fold; the validation portion is only transformed. Final model artifacts are fitted again on the complete labeled training set.

### Preprocessing and feature engineering

1. Replace the missing-value token `./.` with missing values in all panels. Preserve the original panel-specific handling of `AL_` and `EK_` columns; only MASTER's `AL_` columns are explicitly coerced to numeric at the raw-cleaning stage.
2. Drop features with over **85%** missing values, using training-fold statistics only.
3. Add missingness indicators to the panels that use them (CANCER and CFTR).
4. Apply training-derived frequency encoding to `CAT_1`; encode other categorical/amino-acid fields with an unknown-category policy.
5. Apply scaled **KNN imputation** to the relevant categorical branch and **IterativeImputer (Bayesian Ridge)** to the numeric branch. One-hot encoded columns are aligned to the training schema.
6. Retain the top **80%** of processed features by LightGBM importance and apply `RobustScaler` to selected continuous features.

CANCER retains the categorical/missing-mask routing used in the validated final training pipeline. All fitted preprocessing state is stored with each trained model; test data are never used to refit it.

### Panel-specific models

| Panel | Final classifier | Sampling and calibration | Decision policy |
|:--|:--|:--|:--|
| MASTER | Stacking: LightGBM + XGBoost → logistic regression | BorderlineSMOTE inside calibrated estimator; isotonic calibration | EM prevalence estimate with safety fallback and a prior-dependent threshold |
| CANCER | ExtraTrees | BorderlineSMOTE; no isotonic calibration | EM range safety gate, ±0.08 deviation gate and prior-dependent threshold |
| PAH | ExtraTrees | SMOTE and isotonic calibration on resampled data | Fixed announced test prior and fixed OOF-derived threshold |
| CFTR | BalancedRandomForest | Internal class balancing; isotonic calibration | EM prevalence estimate with safety fallback and a prior-dependent threshold |

The fallback/fixed priors use the organizer's approximate published panel distributions. They are policy inputs, not test labels.

## Evaluation

The competition's stated score for a single panel is:

\[
S_{\mathrm{panel}} = \frac{F1 + MCC}{2}
\]

The overall model score is the unweighted arithmetic mean of the four panel scores. The organizer identified PR-AUC as a tie-breaking metric. Panel scores below are from our **internal OOF-based resampling simulation at the announced clinical-stress prevalence**; they are **not final competition test results**.

| Panel | Simulated clinical-stress panel score |
|:--|--:|
| MASTER | 0.434924 |
| CANCER | 0.659077 |
| PAH | 0.482549 |
| CFTR | 0.693259 |
| **Four-panel mean** | **0.567452** |

The simulation reweights/resamples OOF predictions to approximate the disclosed benign-majority test distributions. It cannot reproduce the difficulty or labels of the unseen final test cases. In particular, results from applying a trained model to its own training rows should **not** be interpreted as held-out generalization performance.

The training workflow also produces fold diagnostics, prevalence-policy reports, preprocessing audits, environment metadata and saved model artifacts. Where those files are included in the repository, they should be interpreted as internal experiment records rather than independent clinical validation.

## Repository structure

```text
.
├── Mutation_Prediction.ipynb          # Training, OOF evaluation and final model fitting
├── predict_final.py                  # Offline inference and competition-format JSON output
├── requirements.txt                  # Windows inference dependencies
├── README.md
├── models/                           # Trained panel artifacts, if published
│   ├── MASTER/
│   ├── CANCER/
│   ├── PAH/
│   └── CFTR/
├── reports/                          # Aggregate experiment reports, if published
└── environment/                      # Environment/version metadata, if published
```

**Note:** `predict_final.py` is the competition submission interface; it is not the code that trained the models. The notebook contains the complete final training procedure. Some directories above will only appear when their corresponding artifacts are uploaded.

## Reproducing the training workflow

1. Obtain authorized access to the organizer-provided training CSVs and place the four files in your own Google Drive at the paths listed above, or update the notebook's loading paths locally.
2. Open `Mutation_Prediction.ipynb` in Google Colab and mount Google Drive.
3. Install the package versions compatible with the original training environment; the recorded environment metadata can be used as a reference. The original final training run used Python 3.13.15, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, imbalanced-learn 0.14.2, LightGBM 4.6.0 and XGBoost 3.4.1.
4. Run the notebook's training workflow. It generates OOF reports, final fitted artifacts and a saved package in Google Drive.

The notebook's training code uses fixed random seeds, but exact byte-for-byte predictions across operating systems or different numerical-library builds are **not guaranteed**.

## Running offline inference

On the tested Windows setup, `predict_final.py` loads saved per-panel preprocessing and models, restores MASTER's portable XGBoost model when applicable, reads one **unlabeled CSV per panel**, and writes a UTF-8 JSON file in the competition schema. Place the four CSV files in the input directory configured at the top of the script and set the four input filenames there.

```powershell
.venv\Scripts\python.exe predict_final.py
```

The result is `TEAM_882005_FINAL.json` with the required team metadata and one prediction per variant, including `id`, `panel`, `predicted_class` (`"0"` or `"1"`) and `predicted_prob` (probability of Pathogenic = 1). The test CSVs and the generated competition-test predictions are **not provided in this repository**.

If the repository contains only source code and not pre-trained artifacts, model fitting is required before inference. Trained binary artifacts may also require a compatible dependency stack and platform-specific export, particularly for XGBoost.

## Reproducibility and responsible use

- All learned imputers, encoders, feature selectors and scalers are fitted on training data only and saved as part of the model artifacts.
- The system checks input schemas, variant IDs and the finiteness/range of predicted probabilities before serialization.
- The released source should be inspected for saved notebook outputs and local secrets before each publication.
- No organizer-supplied raw training or final test CSV is redistributed here.
- This is a competition research implementation, not a clinically approved decision-support device.

## Acknowledgment

The team acknowledges the TEKNOFEST 2026 Health AI competition organizers for providing the challenge specifications and training datasets. Dataset descriptions and anticipated class proportions refer to the competition documentation; they do not imply public redistribution rights.
