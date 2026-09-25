# -*- coding: utf-8 -*-
"""
TEKNOFEST 2026 — FINAL PRODUCTION FREEZE V2 (LOCKED)
=====================================================

ONE SCRIPT / TWO INTERNAL STAGES:

A) FINAL LEAKAGE-SAFE OOF
   Raw labeled data
   -> raw normalization/cleaning
   -> fold-local preprocessing
   -> fold-local feature selection/scaling
   -> frozen panel model
   -> OOF probabilities
   -> freeze threshold / prior policy using official score:
          (F1 + MCC) / 2

B) FINAL 100% FIT + SAVE
   The SAME frozen pipeline
   -> fit preprocessing on 100% real labeled training data
   -> fit Top80 + RobustScaler on 100%
   -> fit frozen final model on 100%
   -> in-memory artifact health check
   -> save complete PKL + reports + environment metadata

IMPORTANT
---------
This script is SELF-CONTAINED with respect to the earlier exploratory cells:
- it DOES NOT need:
      datasets = [genel_df, kancer_df, PAH_df, CFTR_df]
- it DOES NOT need the exploratory seaborn/matplotlib cell
- it performs internally EXACTLY as the validated legacy path:
      './.' -> NaN        for ALL panels
      AL_*  -> numeric    for MASTER only
      EK_*  -> unchanged  for ALL panels
      AL_*  -> unchanged  for CANCER / PAH / CFTR
      Label validation -> {0,1}

It still expects the FOUR RAW LABELED DataFrames to already exist in the same
notebook namespace:
    genel_df
    kancer_df
    PAH_df
    CFTR_df

Recommended in Colab/Jupyter:
    %run -i /path/to/TEKNOFEST_FINAL_PRODUCTION_FREEZE_V2.py

FROZEN PANEL DECISIONS
----------------------
MASTER:
    ValuesOnly
    -> Top80 LightGBM
    -> RobustScaler
    -> BorderlineSMOTE
    -> Stacking(LightGBM + XGBoost -> LogisticRegression)
    -> REAL Isotonic calibration
    -> ensemble=False (keep accepted production behavior)
    -> EM + SafetyGate + dynamic threshold map

CANCER:
    Values+Masks
    -> LEGACY CAT/AA mask routing
       (CAT_*_is_missing / AA_*_is_missing enter CAT/AA branch)
    -> Top80 LightGBM
    -> RobustScaler
    -> BorderlineSMOTE
    -> ExtraTrees
    -> NO Isotonic
    -> EM + Range SafetyGate [8%, 35%]
    -> Deviation Gate ±0.08 around official prior 16.67%
    -> dynamic threshold map

PAH:
    ValuesOnly
    -> Top80 LightGBM
    -> RobustScaler
    -> SMOTE(1.0)
    -> ExtraTrees
    -> Isotonic ON RESAMPLED DATA
    -> DEFAULT sklearn ensemble behavior (DO NOT pass ensemble=False)
    -> Fixed prior = 100/350 = 28.5714%
    -> fixed OOF-derived threshold

CFTR:
    Values+Masks
    -> Top80 LightGBM
    -> RobustScaler
    -> BalancedRandomForest (no external SMOTE)
    -> REAL Isotonic calibration
    -> ensemble=False
    -> EM + SafetyGate + dynamic threshold map

DOES NOT DO HERE
----------------
- reload saved PKLs
- Label-less raw test inference
- final JSON serialization
- offline no-internet smoke test

Those are intentionally kept for the NEXT separate inference cell/script.
"""

# =============================================================================
# 0) Determinism / imports
# =============================================================================

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import sys
import json
import hashlib
import platform
import inspect
import shutil
import warnings
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline


# =============================================================================
# 1) Locked global config
# =============================================================================

ARTIFACT_VERSION = "TEKNOFEST_2026_FINAL_V2_LOCKED"

RANDOM_STATE = 42
OUTER_SPLITS = 5
CALIBRATION_CV = 3

MISSING_DROP_THRESHOLD = 85.0
FEATURE_KEEP_FRAC = 0.80

THRESHOLD_GRID = np.arange(0.05, 0.991, 0.01)
THRESHOLD_BOOTSTRAPS = 80

POLICY_EVAL_SEEDS = [
    123, 231, 321, 777, 999,
    1357, 2468, 4321, 8642, 9753
]

EM_MIN_PRIOR = 0.08
EM_MAX_PRIOR = 0.35

# FINAL CANCER LOCK:
# Keep EM dynamic, but prevent the Cancer prior estimate from drifting too far
# from the organizer's expected prior.
CANCER_DEVIATION_GATE = 0.08

THRESHOLD_PRIOR_GRID = np.round(
    np.arange(0.08, 0.3501, 0.01),
    6,
)

# =============================================================================
# DUAL PERSISTENCE LOCK
# Save locally in /content AND persistently in Google Drive.
# Every completed PKL is copied to Drive immediately after it is saved locally.
# At the end, the complete package is synchronized to Drive again.
# =============================================================================

CONTENT_OUTPUT_ROOT = Path("/content/TEKNOFEST_FINAL_PACKAGE_V2")
DRIVE_OUTPUT_ROOT = Path("/content/drive/MyDrive/TEKNOFEST_FINAL_PACKAGE_V2")

# Mount Google Drive automatically when running in Colab.
try:
    from google.colab import drive as _colab_drive

    if not Path("/content/drive/MyDrive").exists():
        _colab_drive.mount(
            "/content/drive"
        )
except Exception as e:
    raise RuntimeError(
        "Google Drive must be available because this final production script "
        "is locked to save every final artifact in BOTH /content and MyDrive. "
        f"Drive mount failed: {e}"
    )

OUTPUT_ROOT = CONTENT_OUTPUT_ROOT
MODELS_DIR = OUTPUT_ROOT / "models"
REPORTS_DIR = OUTPUT_ROOT / "reports"
ENV_DIR = OUTPUT_ROOT / "environment"

DRIVE_MODELS_DIR = DRIVE_OUTPUT_ROOT / "models"
DRIVE_REPORTS_DIR = DRIVE_OUTPUT_ROOT / "reports"
DRIVE_ENV_DIR = DRIVE_OUTPUT_ROOT / "environment"

for d in (
    OUTPUT_ROOT,
    MODELS_DIR,
    REPORTS_DIR,
    ENV_DIR,
    DRIVE_OUTPUT_ROOT,
    DRIVE_MODELS_DIR,
    DRIVE_REPORTS_DIR,
    DRIVE_ENV_DIR,
):
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


PANELS = {
    "MASTER": {
        "df_name": "genel_df",
        "use_missing_masks": False,
        "official_prior": 500 / 3500,
        "prior_policy": "EM_SAFETYNET",
        "model_family": "STACKING_BORDERLINE_REAL_ISOTONIC",
        "calibration_ensemble_mode": "FORCED_FALSE",
    },

    "CANCER": {
        "df_name": "kancer_df",
        "use_missing_masks": True,
        "official_prior": 100 / 600,
        "prior_policy": "EM_SAFETYNET_DEVIATION_GATE",
        "legacy_mask_routing": True,
        "deviation_gate": CANCER_DEVIATION_GATE,
        "model_family": "EXTRATREES_BORDERLINE",
        "calibration_ensemble_mode": "NONE",
    },

    "PAH": {
        "df_name": "PAH_df",
        "use_missing_masks": False,
        "official_prior": 100 / 350,
        "prior_policy": "FIXED_PRIOR",
        "model_family": "EXTRATREES_SMOTE_RESAMPLED_ISOTONIC",
        # IMPORTANT FINAL FIX:
        # omit ensemble argument -> sklearn DEFAULT/AUTO behavior.
        "calibration_ensemble_mode": "DEFAULT",
    },

    "CFTR": {
        "df_name": "CFTR_df",
        "use_missing_masks": True,
        "official_prior": 20 / 120,
        "prior_policy": "EM_SAFETYNET",
        "model_family": "BALANCED_RF_REAL_ISOTONIC",
        # Confirmed winner in final ensemble confirmation.
        "calibration_ensemble_mode": "FORCED_FALSE",
    },
}


# =============================================================================
# 2) Utility helpers
# =============================================================================

def now_utc_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def package_version(name):
    try:
        return importlib_metadata.version(
            name
        )
    except Exception:
        return "UNKNOWN"


def sklearn_ensemble_default():
    try:
        return str(
            inspect.signature(
                CalibratedClassifierCV
            ).parameters["ensemble"].default
        )
    except Exception:
        return "UNKNOWN"


def dataframe_fingerprint(df):
    """
    Fingerprint AFTER the script's own raw normalization.
    """
    h = hashlib.sha256()

    h.update(
        "|".join(
            map(str, df.columns)
        ).encode("utf-8")
    )

    h.update(
        pd.util.hash_pandas_object(
            df,
            index=True,
        ).values.tobytes()
    )

    return h.hexdigest()


# =============================================================================
# 3) SELF-CONTAINED RAW NORMALIZATION
#    Replaces the dependence on the earlier exploratory cells.
# =============================================================================

def normalize_raw_dataframe(
    df_input,
    df_name,
):
    """
    Normalize a RAW labeled panel WITHOUT mutating the original notebook DataFrame.

    IMPORTANT — this reproduces the validated LEGACY cleaning behavior EXACTLY:

    ALL PANELS:
        './.' -> NaN

    MASTER ONLY (genel_df):
        AL_* -> numeric using pd.to_numeric(errors='coerce')

    CANCER / PAH / CFTR:
        AL_* is NOT force-converted here.

    ALL PANELS:
        EK_* is NOT force-converted here.

    Then:
        Label -> numeric integer and validate {0,1}
        verify Variant_ID if present
        remove fully empty rows / rows with missing Label

    We intentionally do NOT:
        - convert EK_* globally
        - convert AL_* globally
        - impute here
        - one-hot encode here
        - scale here
        - create missing masks here

    Those later operations remain inside the leakage-safe fold-local pipeline.
    """
    df = df_input.copy()

    original_shape = tuple(
        df.shape
    )

    # Count explicit "./." tokens before replacement.
    explicit_missing_tokens = 0

    for c in df.columns:
        try:
            explicit_missing_tokens += int(
                df[c]
                .astype(str)
                .str.strip()
                .eq("./.")
                .sum()
            )
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # LEGACY STEP 1 — ALL PANELS:
    # './.' -> NaN
    # -------------------------------------------------------------------------
    df = df.replace(
        r"^\s*\./\.\s*$",
        np.nan,
        regex=True,
    )

    # -------------------------------------------------------------------------
    # LEGACY STEP 2 — MASTER ONLY:
    # AL_* -> numeric
    #
    # DO NOT convert EK_*.
    # DO NOT convert AL_* in CANCER / PAH / CFTR.
    # -------------------------------------------------------------------------
    is_master = (
        df_name == "genel_df"
    )

    master_al_columns = []
    newly_coerced_to_nan = {}

    if is_master:
        master_al_columns = [
            c for c in df.columns
            if str(c).startswith("AL_")
        ]

        for c in master_al_columns:
            before_non_missing = int(
                df[c].notna().sum()
            )

            converted = pd.to_numeric(
                df[c],
                errors="coerce",
            )

            after_non_missing = int(
                converted.notna().sum()
            )

            newly_coerced_to_nan[c] = max(
                0,
                before_non_missing - after_non_missing,
            )

            df[c] = converted

    # Remove completely empty rows first.
    df = df.dropna(
        how="all"
    ).reset_index(
        drop=True
    )

    if "Label" not in df.columns:
        raise ValueError(
            f"{df_name}: Label column missing."
        )

    # Label must be genuinely numerical 0/1.
    label_numeric = pd.to_numeric(
        df["Label"],
        errors="coerce",
    )

    label_invalid_nonempty = (
        df["Label"].notna()
        & label_numeric.isna()
    )

    if label_invalid_nonempty.any():
        examples = (
            df.loc[
                label_invalid_nonempty,
                "Label",
            ]
            .astype(str)
            .unique()
            .tolist()[:10]
        )

        raise ValueError(
            f"{df_name}: non-numeric Label values found: {examples}"
        )

    df["Label"] = label_numeric

    # Drop rows whose Label is actually missing.
    df = df.dropna(
        subset=["Label"]
    ).reset_index(
        drop=True
    )

    # Require integer-equivalent labels.
    if not np.allclose(
        df["Label"].to_numpy(
            dtype=float
        ),
        np.round(
            df["Label"].to_numpy(
                dtype=float
            )
        ),
    ):
        raise ValueError(
            f"{df_name}: Label contains non-integer values."
        )

    df["Label"] = (
        df["Label"]
        .round()
        .astype(int)
    )

    bad_labels = sorted(
        set(
            df["Label"].unique()
        )
        - {0, 1}
    )

    if bad_labels:
        raise ValueError(
            f"{df_name}: invalid labels {bad_labels}; expected only 0/1."
        )

    # Training IDs should be stable and unique.
    if "Variant_ID" in df.columns:
        if df["Variant_ID"].isna().any():
            raise ValueError(
                f"{df_name}: Variant_ID contains missing values."
            )

        if df["Variant_ID"].astype(str).duplicated().any():
            dup = (
                df.loc[
                    df["Variant_ID"]
                    .astype(str)
                    .duplicated(
                        keep=False
                    ),
                    "Variant_ID",
                ]
                .astype(str)
                .unique()
                .tolist()[:10]
            )

            raise ValueError(
                f"{df_name}: duplicate Variant_ID values found: {dup}"
            )

    total_new_numeric_nan = int(
        sum(
            newly_coerced_to_nan.values()
        )
    )

    changed_numeric_cols = {
        c: n
        for c, n in newly_coerced_to_nan.items()
        if n > 0
    }

    report = {
        "df_name": df_name,
        "original_rows": int(
            original_shape[0]
        ),
        "original_columns": int(
            original_shape[1]
        ),
        "normalized_rows": int(
            df.shape[0]
        ),
        "normalized_columns": int(
            df.shape[1]
        ),
        "explicit_dot_slash_dot_tokens_replaced": int(
            explicit_missing_tokens
        ),

        # Legacy cleaning audit fields.
        "legacy_cleaning_mode": True,
        "dot_slash_dot_replaced_all_panels": True,
        "master_only_al_numeric_conversion": bool(
            is_master
        ),
        "master_al_columns_checked": int(
            len(master_al_columns)
        ),
        "ek_numeric_conversion_applied": False,
        "global_al_numeric_conversion_applied": False,

        "new_nan_from_numeric_coercion_total": total_new_numeric_nan,
        "new_nan_from_numeric_coercion_by_column": changed_numeric_cols,
        "pathogenic_count": int(
            (df["Label"] == 1).sum()
        ),
        "benign_count": int(
            (df["Label"] == 0).sum()
        ),
        "pathogenic_prior": float(
            df["Label"].mean()
        ),
    }

    return df, report

def get_dataframe(name):
    """
    Get a raw DataFrame from the current notebook/script namespace,
    then normalize it internally.
    """
    if name not in globals():
        raise RuntimeError(
            f"{name} is not loaded in memory.\n"
            f"Run this script in the same notebook namespace, e.g.:\n"
            f"    %run -i /path/to/TEKNOFEST_FINAL_PRODUCTION_FREEZE_V2.py"
        )

    return normalize_raw_dataframe(
        globals()[name],
        name,
    )


# =============================================================================
# 4) Metrics
# =============================================================================

def safe_ap(
    y,
    p,
    positive=1,
):
    y = np.asarray(y)
    p = np.asarray(p)

    if positive == 1:
        return average_precision_score(
            y,
            p,
        )

    return average_precision_score(
        (y == 0).astype(int),
        1.0 - p,
    )


def safe_roc(
    y,
    p,
):
    y = np.asarray(y)

    if len(
        np.unique(y)
    ) != 2:
        return np.nan

    return roc_auc_score(
        y,
        p,
    )


def probability_metrics(
    y,
    p,
):
    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        1e-6,
        1 - 1e-6,
    )

    return {
        "pr_auc_pathogenic": float(
            safe_ap(
                y,
                p,
                1,
            )
        ),
        "pr_auc_benign": float(
            safe_ap(
                y,
                p,
                0,
            )
        ),
        "roc_auc": float(
            safe_roc(
                y,
                p,
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                p,
            )
        ),
        "logloss": float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),
    }


def official_binary_metrics(
    y,
    pred,
):
    f1 = float(
        f1_score(
            y,
            pred,
            zero_division=0,
        )
    )

    mcc = float(
        matthews_corrcoef(
            y,
            pred,
        )
    )

    return {
        "f1": f1,
        "mcc": mcc,
        "panel_score_f1_mcc": float(
            (f1 + mcc) / 2.0
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                pred,
            )
        ),
    }


# =============================================================================
# 5) Calibrator constructor
#    CRITICAL FINAL PAH/CFTR FIX IS HERE.
# =============================================================================

def make_calibrator(
    estimator,
    cv=CALIBRATION_CV,
    method="isotonic",
    ensemble_mode="FORCED_FALSE",
):
    """
    ensemble_mode:

    "DEFAULT"
        DO NOT pass `ensemble` at all.
        This reproduces sklearn's default/auto behavior.
        FINAL PAH uses this.

    "FORCED_FALSE"
        explicitly pass ensemble=False.
        FINAL CFTR uses this.
        MASTER keeps its accepted production behavior.

    Never use this for CANCER.
    """
    if ensemble_mode == "DEFAULT":
        try:
            return CalibratedClassifierCV(
                estimator=estimator,
                method=method,
                cv=cv,
            )
        except TypeError:
            return CalibratedClassifierCV(
                base_estimator=estimator,
                method=method,
                cv=cv,
            )

    if ensemble_mode == "FORCED_FALSE":
        kwargs = {
            "method": method,
            "cv": cv,
            "ensemble": False,
        }

        try:
            return CalibratedClassifierCV(
                estimator=estimator,
                **kwargs,
            )
        except TypeError:
            return CalibratedClassifierCV(
                base_estimator=estimator,
                **kwargs,
            )

    raise ValueError(
        f"Unsupported ensemble_mode={ensemble_mode}"
    )


# =============================================================================
# 6) Leakage-safe preprocessing
# =============================================================================

def _make_knn_imputer():
    """
    Preserve feature count even if an unusual fold contains an all-missing
    KNN-column, when supported by the installed sklearn.
    """
    try:
        return KNNImputer(
            n_neighbors=5,
            keep_empty_features=True,
        )
    except TypeError:
        return KNNImputer(
            n_neighbors=5,
        )


def _make_mice_imputer():
    kwargs = {
        "estimator": BayesianRidge(),
        "max_iter": 10,
        "random_state": RANDOM_STATE,
    }

    try:
        return IterativeImputer(
            keep_empty_features=True,
            **kwargs,
        )
    except TypeError:
        return IterativeImputer(
            **kwargs,
        )


def fit_preprocessor(
    X_train_raw,
    add_missing_indicators,
    panel=None,
):
    X = X_train_raw.copy()

    legacy_mask_routing = bool(
        panel == "CANCER"
        and add_missing_indicators
    )

    state = {
        "panel": panel,
        "add_missing_indicators": bool(
            add_missing_indicators
        ),
        "legacy_mask_routing": legacy_mask_routing,
        "missing_drop_threshold_pct": float(
            MISSING_DROP_THRESHOLD
        ),
    }

    # -------------------------------------------------------------------------
    # 6.1 Drop >85% missing: FIT ON TRAIN ONLY
    # -------------------------------------------------------------------------
    missing_pct = (
        X.isna().mean()
        * 100.0
    )

    drop_cols = missing_pct[
        missing_pct
        > MISSING_DROP_THRESHOLD
    ].index.tolist()

    X = X.drop(
        columns=drop_cols,
        errors="ignore",
    )

    state["dropped_columns"] = drop_cols
    state["kept_base_columns"] = X.columns.tolist()

    # -------------------------------------------------------------------------
    # 6.2 Missingness masks: panel-specific frozen decision
    # -------------------------------------------------------------------------
    missing_sources = [
        c for c in X.columns
        if X[c].isna().any()
    ]

    state[
        "missing_sources"
    ] = missing_sources

    if add_missing_indicators:
        for c in missing_sources:
            X[
                f"{c}_is_missing"
            ] = X[c].isna().astype(
                np.int8
            )

    # -------------------------------------------------------------------------
    # 6.3 CAT_1 frequency encoding: FIT ON TRAIN ONLY
    # -------------------------------------------------------------------------
    if "CAT_1" in X.columns:
        freq = X[
            "CAT_1"
        ].value_counts(
            normalize=True,
            dropna=True,
        )

        state[
            "cat1_frequency_map"
        ] = freq

        X["CAT_1"] = X[
            "CAT_1"
        ].map(
            freq
        )

    # -------------------------------------------------------------------------
    # 6.4 Other CAT/AA -> temporary ordinal encoding
    # -------------------------------------------------------------------------
    # FINAL CANCER FIX:
    # In the validated legacy Cancer path, masks whose names start with
    # CAT / AA also entered the temporary categorical branch.
    # Other panels keep the V2 behavior and exclude *_is_missing masks here.
    cat_aa = [
        c for c in X.columns
        if (
            str(c).startswith(
                "CAT"
            )
            or str(c).startswith(
                "AA"
            )
        )
        and c != "CAT_1"
        and (
            legacy_mask_routing
            or not str(c).endswith(
                "_is_missing"
            )
        )
    ]

    state[
        "cat_aa_columns"
    ] = cat_aa

    state[
        "cat_aa_missing_mask_columns"
    ] = [
        c for c in cat_aa
        if str(c).endswith(
            "_is_missing"
        )
    ]

    if cat_aa:
        for c in cat_aa:
            X[c] = X[c].astype(
                object
            )

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )

        X[
            cat_aa
        ] = encoder.fit_transform(
            X[
                cat_aa
            ]
        )

        state[
            "ordinal_encoder"
        ] = encoder

    # -------------------------------------------------------------------------
    # 6.5 KNN path: CAT_1 + encoded CAT/AA
    # -------------------------------------------------------------------------
    knn_cols = [
        c for c in (
            ["CAT_1"]
            + cat_aa
        )
        if c in X.columns
    ]

    state[
        "knn_columns"
    ] = knn_cols

    if knn_cols:
        knn_scaler = MinMaxScaler()

        Z = knn_scaler.fit_transform(
            X[
                knn_cols
            ]
        )

        knn_imputer = _make_knn_imputer()

        Z = knn_imputer.fit_transform(
            Z
        )

        # Safety: shape must be preserved.
        if Z.shape[1] != len(
            knn_cols
        ):
            raise RuntimeError(
                "KNNImputer changed the number of columns. "
                "An all-missing KNN feature likely exists in this fold."
            )

        X[
            knn_cols
        ] = knn_scaler.inverse_transform(
            Z
        )

        if cat_aa:
            X[
                cat_aa
            ] = X[
                cat_aa
            ].round()

        state[
            "knn_scaler"
        ] = knn_scaler

        state[
            "knn_imputer"
        ] = knn_imputer

    # -------------------------------------------------------------------------
    # 6.6 MICE numeric path
    # -------------------------------------------------------------------------
    numeric_cols = [
        c for c in X.columns
        if (
            c not in knn_cols
            and pd.api.types.is_numeric_dtype(
                X[c]
            )
            and not str(c).endswith(
                "_is_missing"
            )
        )
    ]

    state[
        "numeric_columns"
    ] = numeric_cols

    if numeric_cols:
        mice = _make_mice_imputer()

        Znum = mice.fit_transform(
            X[
                numeric_cols
            ]
        )

        if Znum.shape[1] != len(
            numeric_cols
        ):
            raise RuntimeError(
                "IterativeImputer changed the number of columns. "
                "An all-missing numeric feature likely exists in this fold."
            )

        X[
            numeric_cols
        ] = Znum

        X[
            numeric_cols
        ] = X[
            numeric_cols
        ].clip(
            lower=0
        )

        state[
            "mice_imputer"
        ] = mice

    # -------------------------------------------------------------------------
    # 6.7 Final one-hot schema on TRAIN
    # -------------------------------------------------------------------------
    if cat_aa:
        X = pd.get_dummies(
            X,
            columns=cat_aa,
            drop_first=True,
        )

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(
                np.int8
            )

    state[
        "final_preprocessed_columns"
    ] = X.columns.tolist()

    return (
        X.astype(float),
        state,
    )


def transform_preprocessor(
    X_raw,
    state,
):
    X = X_raw.copy()

    # Restore EXACT training base schema.
    for c in state[
        "kept_base_columns"
    ]:
        if c not in X.columns:
            X[c] = np.nan

    X = X[
        state[
            "kept_base_columns"
        ]
    ].copy()

    # Missing masks must reflect THIS sample's raw missingness.
    if state[
        "add_missing_indicators"
    ]:
        for c in state[
            "missing_sources"
        ]:
            X[
                f"{c}_is_missing"
            ] = X[
                c
            ].isna().astype(
                np.int8
            )

    # CAT_1 using TRAIN frequency map.
    if (
        "cat1_frequency_map"
        in state
        and "CAT_1" in X.columns
    ):
        X[
            "CAT_1"
        ] = X[
            "CAT_1"
        ].map(
            state[
                "cat1_frequency_map"
            ]
        )

    cat_aa = state.get(
        "cat_aa_columns",
        [],
    )

    # Cast to object before transform to keep train/test dtype behavior aligned.
    if cat_aa:
        for c in cat_aa:
            X[c] = X[c].astype(
                object
            )

        X[
            cat_aa
        ] = state[
            "ordinal_encoder"
        ].transform(
            X[
                cat_aa
            ]
        )

    knn_cols = state.get(
        "knn_columns",
        [],
    )

    if knn_cols:
        Z = state[
            "knn_scaler"
        ].transform(
            X[
                knn_cols
            ]
        )

        Z = state[
            "knn_imputer"
        ].transform(
            Z
        )

        if Z.shape[1] != len(
            knn_cols
        ):
            raise RuntimeError(
                "KNN transform changed column count."
            )

        X[
            knn_cols
        ] = state[
            "knn_scaler"
        ].inverse_transform(
            Z
        )

        if cat_aa:
            X[
                cat_aa
            ] = X[
                cat_aa
            ].round()

    numeric_cols = state.get(
        "numeric_columns",
        [],
    )

    if numeric_cols:
        Znum = state[
            "mice_imputer"
        ].transform(
            X[
                numeric_cols
            ]
        )

        if Znum.shape[1] != len(
            numeric_cols
        ):
            raise RuntimeError(
                "MICE transform changed column count."
            )

        X[
            numeric_cols
        ] = Znum

        X[
            numeric_cols
        ] = X[
            numeric_cols
        ].clip(
            lower=0
        )

    # IMPORTANT:
    # Training used drop_first=True.
    # At transform we create ALL available dummies, then reindex to the exact
    # TRAIN schema. This is safe for unseen/missing category combinations.
    if cat_aa:
        X = pd.get_dummies(
            X,
            columns=cat_aa,
            drop_first=False,
        )

    X = X.reindex(
        columns=state[
            "final_preprocessed_columns"
        ],
        fill_value=0,
    )

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(
                np.int8
            )

    out = X.astype(float)

    if not np.isfinite(
        out.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Non-finite value remained after preprocessing transform."
        )

    return out


# =============================================================================
# 7) Frozen Top80 feature stage + RobustScaler
# =============================================================================

def fit_feature_stage(
    X_train,
    y_train,
):
    selector = LGBMClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=1,
    )

    selector.fit(
        X_train,
        y_train,
    )

    n_keep = max(
        1,
        int(
            len(
                X_train.columns
            )
            * FEATURE_KEEP_FRAC
        ),
    )

    selected = (
        pd.Series(
            selector.feature_importances_,
            index=X_train.columns,
        )
        .sort_values(
            ascending=False
        )
        .head(
            n_keep
        )
        .index
        .tolist()
    )

    Xs = X_train[
        selected
    ].copy()

    continuous = [
        c for c in selected
        if (
            Xs[c].nunique(
                dropna=True
            )
            > 2
            and not str(c).endswith(
                "_is_missing"
            )
        )
    ]

    scaler = None

    if continuous:
        scaler = RobustScaler()

        Xs[
            continuous
        ] = scaler.fit_transform(
            Xs[
                continuous
            ]
        )

    state = {
        "feature_keep_fraction": float(
            FEATURE_KEEP_FRAC
        ),
        "selected_features": selected,
        "continuous_features": continuous,
        "robust_scaler": scaler,
        "selector_model": selector,
    }

    return (
        Xs,
        state,
    )


def transform_feature_stage(
    X,
    state,
):
    Xs = X.reindex(
        columns=state[
            "selected_features"
        ],
        fill_value=0,
    ).copy()

    if (
        state[
            "robust_scaler"
        ]
        is not None
        and state[
            "continuous_features"
        ]
    ):
        Xs[
            state[
                "continuous_features"
            ]
        ] = state[
            "robust_scaler"
        ].transform(
            Xs[
                state[
                    "continuous_features"
                ]
            ]
        )

    if not np.isfinite(
        Xs.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Non-finite value remained after feature stage."
        )

    return Xs


# =============================================================================
# 8) Frozen sampling / model builders
# =============================================================================

def k_neighbors_for_sampling(
    y,
):
    minority = int(
        pd.Series(
            y
        ).value_counts().min()
    )

    return max(
        1,
        min(
            5,
            minority - 1,
        ),
    )


def make_smote(
    y,
):
    return SMOTE(
        sampling_strategy=1.0,
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors_for_sampling(
            y
        ),
    )


def make_borderline_smote(
    y,
):
    return BorderlineSMOTE(
        sampling_strategy=1.0,
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors_for_sampling(
            y
        ),
    )


def make_master_stacking():
    return StackingClassifier(
        estimators=[
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=6,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    n_jobs=1,
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=1,
                ),
            ),
        ],
        final_estimator=LogisticRegression(
            max_iter=1000
        ),
        cv=3,
        n_jobs=1,
    )


def make_extra_trees():
    return ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def make_cftr_balanced_rf():
    return BalancedRandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
    )


# =============================================================================
# 9) FINAL FROZEN panel model fit
# =============================================================================

def fit_frozen_model(
    panel,
    X_train,
    y_train,
):
    # -------------------------------------------------------------------------
    # MASTER
    # -------------------------------------------------------------------------
    if panel == "MASTER":
        base = ImbPipeline([
            (
                "borderline_smote",
                make_borderline_smote(
                    y_train
                ),
            ),
            (
                "stacking",
                make_master_stacking(),
            ),
        ])

        model = make_calibrator(
            base,
            method="isotonic",
            cv=CALIBRATION_CV,
            ensemble_mode="FORCED_FALSE",
        )

        model.fit(
            X_train,
            y_train,
        )

        return (
            model,
            float(
                np.mean(
                    y_train
                )
            ),
            {
                "external_sampling": "BorderlineSMOTE inside calibrated estimator",
                "calibration": "isotonic_REAL_out_of_resampling",
                "calibration_cv": CALIBRATION_CV,
                "calibration_ensemble_mode": "FORCED_FALSE",
            },
        )

    # -------------------------------------------------------------------------
    # CANCER
    # -------------------------------------------------------------------------
    if panel == "CANCER":
        smote = make_borderline_smote(
            y_train
        )

        Xr, yr = smote.fit_resample(
            X_train,
            y_train,
        )

        model = make_extra_trees()

        model.fit(
            Xr,
            yr,
        )

        return (
            model,
            float(
                np.mean(
                    yr
                )
            ),
            {
                "external_sampling": "BorderlineSMOTE_1.0",
                "rows_before_sampling": int(
                    len(
                        y_train
                    )
                ),
                "rows_after_sampling": int(
                    len(
                        yr
                    )
                ),
                "calibration": "none",
                "calibration_ensemble_mode": "NONE",
            },
        )

    # -------------------------------------------------------------------------
    # PAH — FINAL FIX:
    # DEFAULT calibrator ensemble behavior (omit ensemble argument)
    # -------------------------------------------------------------------------
    if panel == "PAH":
        smote = make_smote(
            y_train
        )

        Xr, yr = smote.fit_resample(
            X_train,
            y_train,
        )

        model = make_calibrator(
            make_extra_trees(),
            method="isotonic",
            cv=CALIBRATION_CV,
            ensemble_mode="DEFAULT",
        )

        model.fit(
            Xr,
            yr,
        )

        return (
            model,
            float(
                np.mean(
                    yr
                )
            ),
            {
                "external_sampling": "SMOTE_1.0",
                "rows_before_sampling": int(
                    len(
                        y_train
                    )
                ),
                "rows_after_sampling": int(
                    len(
                        yr
                    )
                ),
                "calibration": "isotonic_on_RESAMPLED_data",
                "calibration_cv": CALIBRATION_CV,
                "calibration_ensemble_mode": "DEFAULT",
                "installed_sklearn_default_ensemble": sklearn_ensemble_default(),
            },
        )

    # -------------------------------------------------------------------------
    # CFTR — FINAL CONFIRMED:
    # BalancedRF + REAL isotonic + ensemble=False
    # -------------------------------------------------------------------------
    if panel == "CFTR":
        model = make_calibrator(
            make_cftr_balanced_rf(),
            method="isotonic",
            cv=CALIBRATION_CV,
            ensemble_mode="FORCED_FALSE",
        )

        model.fit(
            X_train,
            y_train,
        )

        return (
            model,
            float(
                np.mean(
                    y_train
                )
            ),
            {
                "external_sampling": "none",
                "internal_balancing": "BalancedRandomForest",
                "calibration": "isotonic_REAL_data",
                "calibration_cv": CALIBRATION_CV,
                "calibration_ensemble_mode": "FORCED_FALSE",
            },
        )

    raise ValueError(
        panel
    )


# =============================================================================
# 10) Final leakage-safe OOF
# =============================================================================

def generate_final_oof(
    panel,
    raw_df,
    use_missing_masks,
):
    y = (
        raw_df[
            "Label"
        ]
        .astype(int)
        .reset_index(
            drop=True
        )
    )

    Xraw = (
        raw_df.drop(
            columns=[
                "Label",
                "Variant_ID",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    for fold, (
        tr,
        va,
    ) in enumerate(
        cv.split(
            Xraw,
            y,
        ),
        start=1,
    ):
        ytr = y.iloc[
            tr
        ]

        yva = y.iloc[
            va
        ]

        # Fold-local preprocessing FIT.
        Xtr_pre, prep_state = fit_preprocessor(
            Xraw.iloc[
                tr
            ].copy(),
            add_missing_indicators=use_missing_masks,
            panel=panel,
        )

        # Validation transform ONLY.
        Xva_pre = transform_preprocessor(
            Xraw.iloc[
                va
            ].copy(),
            prep_state,
        )

        # Fold-local Top80 + scaler.
        Xtr, feature_state = fit_feature_stage(
            Xtr_pre,
            ytr,
        )

        Xva = transform_feature_stage(
            Xva_pre,
            feature_state,
        )

        # Exact frozen panel model.
        model, calibration_prior_fold, train_info = fit_frozen_model(
            panel,
            Xtr,
            ytr,
        )

        pva = model.predict_proba(
            Xva
        )[:, 1]

        if not np.isfinite(
            pva
        ).all():
            raise RuntimeError(
                f"{panel} fold {fold}: non-finite OOF probability."
            )

        if (
            (pva < 0).any()
            or (pva > 1).any()
        ):
            raise RuntimeError(
                f"{panel} fold {fold}: probability outside [0,1]."
            )

        oof[
            va
        ] = pva

        fold_rows.append({
            "panel": panel,
            "fold": fold,
            "train_rows": int(
                len(
                    tr
                )
            ),
            "val_rows": int(
                len(
                    va
                )
            ),
            "train_prior_real": float(
                ytr.mean()
            ),
            "calibration_prior_fold": float(
                calibration_prior_fold
            ),
            "calibration_ensemble_mode": train_info.get(
                "calibration_ensemble_mode",
                "NONE",
            ),
            "features_after_preprocess": int(
                Xtr_pre.shape[1]
            ),
            "selected_features": int(
                Xtr.shape[1]
            ),
            "selected_masks": int(
                sum(
                    str(c).endswith(
                        "_is_missing"
                    )
                    for c in Xtr.columns
                )
            ),
            "val_pr_auc_pathogenic": float(
                safe_ap(
                    yva,
                    pva,
                    1,
                )
            ),
            "val_pr_auc_benign": float(
                safe_ap(
                    yva,
                    pva,
                    0,
                )
            ),
            "val_roc_auc": float(
                safe_roc(
                    yva,
                    pva,
                )
            ),
        })

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            f"{panel}: NaN remained in OOF predictions."
        )

    # Probability prior used by EM:
    # MASTER / CFTR -> real-calibration prior
    # CANCER -> balanced 0.5 anchor
    # PAH -> balanced 0.5 anchor, though PAH does NOT use EM
    if panel in (
        "MASTER",
        "CFTR",
    ):
        calibration_prior = float(
            y.mean()
        )
    else:
        calibration_prior = 0.5

    return {
        "y": y.to_numpy(),
        "p": oof,
        "calibration_prior": calibration_prior,
        "fold_diagnostics": pd.DataFrame(
            fold_rows
        ),
        "probability_metrics": probability_metrics(
            y,
            oof,
        ),
    }


# =============================================================================
# 11) Prior shift / threshold helpers
# =============================================================================

def estimate_test_prior_em(
    calibration_prior,
    test_proba,
    max_iter=100,
    tol=1e-6,
):
    q = np.asarray(
        test_proba,
        dtype=float,
    )

    p0 = float(
        np.clip(
            calibration_prior,
            1e-6,
            1 - 1e-6,
        )
    )

    p = p0

    for _ in range(
        max_iter
    ):
        num = q * (
            p / p0
        )

        den = (
            num
            + (1 - q)
            * (
                (1 - p)
                / (1 - p0)
            )
        )

        post = num / np.clip(
            den,
            1e-12,
            None,
        )

        new_p = float(
            post.mean()
        )

        if abs(
            new_p - p
        ) < tol:
            p = new_p
            break

        p = new_p

    return float(
        np.clip(
            p,
            0.03,
            0.97,
        )
    )


def em_is_reliable(
    value,
):
    return bool(
        np.isfinite(
            value
        )
        and value > 0.05
        and value < 0.95
        and EM_MIN_PRIOR
        <= value
        <= EM_MAX_PRIOR
    )


def choose_production_prior(
    panel,
    em_value,
    official_prior,
):
    """
    Unified production prior decision.

    MASTER / CFTR:
        existing EM + range SafetyGate behavior.

    CANCER:
        existing range SafetyGate, then clip accepted EM to:
            official_prior ± CANCER_DEVIATION_GATE

        With official prior 0.166667 and gate 0.08:
            allowed dynamic interval ~= [0.086667, 0.246667]

    PAH:
        handled separately as FIXED_PRIOR.
    """
    reliable = em_is_reliable(
        em_value
    )

    if not reliable:
        return (
            float(
                official_prior
            ),
            False,
            "FALLBACK_RANGE",
        )

    if panel != "CANCER":
        return (
            float(
                em_value
            ),
            True,
            "EM_ACCEPTED",
        )

    lower = max(
        EM_MIN_PRIOR,
        official_prior
        - CANCER_DEVIATION_GATE,
    )

    upper = min(
        EM_MAX_PRIOR,
        official_prior
        + CANCER_DEVIATION_GATE,
    )

    chosen = float(
        np.clip(
            em_value,
            lower,
            upper,
        )
    )

    status = (
        "EM_ACCEPTED"
        if abs(
            chosen
            - em_value
        ) < 1e-12
        else "EM_CLIPPED_DEVIATION"
    )

    return (
        chosen,
        True,
        status,
    )


def stress_resample(
    y,
    p,
    target_prior,
    seed,
):
    rng = np.random.RandomState(
        seed
    )

    y = np.asarray(
        y
    )

    p = np.asarray(
        p
    )

    pos = np.where(
        y == 1
    )[0]

    neg = np.where(
        y == 0
    )[0]

    current_prior = (
        len(
            pos
        )
        / len(
            y
        )
    )

    if current_prior > target_prior:
        n_pos_target = int(
            round(
                target_prior
                * len(
                    neg
                )
                / (
                    1
                    - target_prior
                )
            )
        )

        n_pos_target = max(
            1,
            min(
                n_pos_target,
                len(
                    pos
                ),
            ),
        )

        selected_pos = rng.choice(
            pos,
            n_pos_target,
            replace=False,
        )

        selected_neg = neg

    else:
        n_neg_target = int(
            round(
                len(
                    pos
                )
                * (
                    1
                    - target_prior
                )
                / target_prior
            )
        )

        n_neg_target = max(
            1,
            min(
                n_neg_target,
                len(
                    neg
                ),
            ),
        )

        selected_pos = pos

        selected_neg = rng.choice(
            neg,
            n_neg_target,
            replace=False,
        )

    idx = np.concatenate([
        selected_pos,
        selected_neg,
    ])

    rng.shuffle(
        idx
    )

    return (
        y[
            idx
        ],
        p[
            idx
        ],
    )


def competition_score_from_predictions(
    y_true,
    y_pred,
):
    m = official_binary_metrics(
        y_true,
        y_pred,
    )

    return float(
        m[
            "panel_score_f1_mcc"
        ]
    )


def best_threshold_for_prior(
    y,
    p,
    target_prior,
    base_seed,
):
    winners = []

    for b in range(
        THRESHOLD_BOOTSTRAPS
    ):
        ys, ps = stress_resample(
            y,
            p,
            target_prior,
            base_seed
            + b,
        )

        scores = []

        for threshold in THRESHOLD_GRID:
            pred = (
                ps
                >= threshold
            ).astype(
                int
            )

            scores.append(
                competition_score_from_predictions(
                    ys,
                    pred,
                )
            )

        winners.append(
            float(
                THRESHOLD_GRID[
                    int(
                        np.argmax(
                            scores
                        )
                    )
                ]
            )
        )

    return float(
        np.median(
            winners
        )
    )


# =============================================================================
# 12) Freeze threshold / prior policy from final OOF
# =============================================================================

def build_threshold_policy(
    panel,
    y,
    p,
    official_prior,
):
    # PAH: fixed official prior, no EM.
    if panel == "PAH":
        threshold = best_threshold_for_prior(
            y,
            p,
            official_prior,
            base_seed=9000
            + int(
                official_prior
                * 100000
            ),
        )

        return {
            "policy": "FIXED_PRIOR",
            "fixed_prior": float(
                official_prior
            ),
            "fixed_threshold": float(
                threshold
            ),
            "optimization_metric": "(F1+MCC)/2",
        }

    # MASTER / CANCER / CFTR:
    # precompute dynamic threshold map.
    priors = list(
        map(
            float,
            THRESHOLD_PRIOR_GRID,
        )
    )

    if not any(
        abs(
            x
            - official_prior
        )
        < 1e-12
        for x in priors
    ):
        priors.append(
            float(
                official_prior
            )
        )

    priors = sorted(
        set(
            round(
                x,
                6,
            )
            for x in priors
        )
    )

    threshold_map = {}

    for prior in priors:
        threshold_map[
            f"{prior:.6f}"
        ] = best_threshold_for_prior(
            y,
            p,
            prior,
            base_seed=7000
            + int(
                prior
                * 100000
            ),
        )

    state = {
        "policy": (
            "EM_SAFETYNET_DEVIATION_GATE"
            if panel == "CANCER"
            else "EM_SAFETYNET"
        ),
        "fallback_prior": float(
            official_prior
        ),
        "em_min_prior": float(
            EM_MIN_PRIOR
        ),
        "em_max_prior": float(
            EM_MAX_PRIOR
        ),
        "threshold_prior_grid": priors,
        "threshold_map": threshold_map,
        "threshold_lookup": "nearest_prior_grid",
        "optimization_metric": "(F1+MCC)/2",
    }

    if panel == "CANCER":
        state[
            "deviation_gate"
        ] = float(
            CANCER_DEVIATION_GATE
        )

        state[
            "deviation_center"
        ] = float(
            official_prior
        )

        state[
            "deviation_lower"
        ] = float(
            max(
                EM_MIN_PRIOR,
                official_prior
                - CANCER_DEVIATION_GATE,
            )
        )

        state[
            "deviation_upper"
        ] = float(
            min(
                EM_MAX_PRIOR,
                official_prior
                + CANCER_DEVIATION_GATE,
            )
        )

    return state


def lookup_threshold(
    policy_state,
    chosen_prior,
):
    if policy_state[
        "policy"
    ] == "FIXED_PRIOR":
        return float(
            policy_state[
                "fixed_threshold"
            ]
        )

    grid = np.asarray(
        policy_state[
            "threshold_prior_grid"
        ],
        dtype=float,
    )

    nearest = float(
        grid[
            np.argmin(
                np.abs(
                    grid
                    - chosen_prior
                )
            )
        ]
    )

    return float(
        policy_state[
            "threshold_map"
        ][
            f"{nearest:.6f}"
        ]
    )


def stress_prior_grid(
    official_prior,
):
    return [
        float(
            np.clip(
                official_prior - 0.05,
                EM_MIN_PRIOR,
                EM_MAX_PRIOR,
            )
        ),
        float(
            np.clip(
                official_prior - 0.025,
                EM_MIN_PRIOR,
                EM_MAX_PRIOR,
            )
        ),
        float(
            official_prior
        ),
        float(
            np.clip(
                official_prior + 0.025,
                EM_MIN_PRIOR,
                EM_MAX_PRIOR,
            )
        ),
        float(
            np.clip(
                official_prior + 0.05,
                EM_MIN_PRIOR,
                EM_MAX_PRIOR,
            )
        ),
    ]


def evaluate_final_policy(
    panel,
    y,
    p,
    calibration_prior,
    policy_state,
    official_prior,
):
    rows = []

    for true_prior in stress_prior_grid(
        official_prior
    ):
        rep_rows = []

        for seed in POLICY_EVAL_SEEDS:
            ys, ps = stress_resample(
                y,
                p,
                true_prior,
                seed,
            )

            if panel == "PAH":
                em = np.nan
                reliable = False
                chosen_prior = official_prior
                gate_status = "FIXED_PRIOR"
            else:
                em = estimate_test_prior_em(
                    calibration_prior,
                    ps,
                )

                (
                    chosen_prior,
                    reliable,
                    gate_status,
                ) = choose_production_prior(
                    panel,
                    em,
                    official_prior,
                )

            threshold = lookup_threshold(
                policy_state,
                chosen_prior,
            )

            pred = (
                ps
                >= threshold
            ).astype(
                int
            )

            metrics = official_binary_metrics(
                ys,
                pred,
            )

            rep_rows.append({
                "em": em,
                "reliable": reliable,
                "gate_status": gate_status,
                "chosen_prior": chosen_prior,
                "threshold": threshold,
                **metrics,
            })

        d = pd.DataFrame(
            rep_rows
        )

        rows.append({
            "panel": panel,
            "true_prior": float(
                true_prior
            ),
            "production_threshold_median": float(
                d[
                    "threshold"
                ].median()
            ),
            "production_f1_mean": float(
                d[
                    "f1"
                ].mean()
            ),
            "production_mcc_mean": float(
                d[
                    "mcc"
                ].mean()
            ),
            "production_panel_score_mean": float(
                d[
                    "panel_score_f1_mcc"
                ].mean()
            ),
            "production_balanced_accuracy_mean": float(
                d[
                    "balanced_accuracy"
                ].mean()
            ),
            "em_mean": (
                float(
                    d[
                        "em"
                    ].mean()
                )
                if panel != "PAH"
                else np.nan
            ),
            "em_mae": (
                float(
                    np.nanmean(
                        np.abs(
                            d[
                                "em"
                            ]
                            - true_prior
                        )
                    )
                )
                if panel != "PAH"
                else np.nan
            ),
            "em_reliable_rate": (
                float(
                    d[
                        "reliable"
                    ].mean()
                )
                if panel != "PAH"
                else np.nan
            ),
            "chosen_prior_mean": (
                float(
                    d[
                        "chosen_prior"
                    ].mean()
                )
                if panel != "PAH"
                else float(
                    official_prior
                )
            ),
            "gate_clip_or_fallback_rate": (
                float(
                    d[
                        "gate_status"
                    ].isin(
                        [
                            "FALLBACK_RANGE",
                            "EM_CLIPPED_DEVIATION",
                        ]
                    ).mean()
                )
                if panel != "PAH"
                else 0.0
            ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 13) Fit COMPLETE final artifact on 100% real labeled data
# =============================================================================

def fit_final_100_percent_artifact(
    panel,
    raw_df,
    raw_normalization_report,
    config,
    oof_result,
    threshold_policy,
):
    y = (
        raw_df[
            "Label"
        ]
        .astype(int)
        .reset_index(
            drop=True
        )
    )

    Xraw = (
        raw_df.drop(
            columns=[
                "Label",
                "Variant_ID",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # 100% preprocessing fit
    # ---------------------------------------------------------
    Xpre, prep_state = fit_preprocessor(
        Xraw,
        add_missing_indicators=config[
            "use_missing_masks"
        ],
        panel=panel,
    )

    # ---------------------------------------------------------
    # 100% Top80 + scaler fit
    # ---------------------------------------------------------
    Xfinal, feature_state = fit_feature_stage(
        Xpre,
        y,
    )

    # ---------------------------------------------------------
    # 100% final model fit
    # ---------------------------------------------------------
    model, calibration_prior, train_info = fit_frozen_model(
        panel,
        Xfinal,
        y,
    )

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": now_utc_iso(),

        "panel": panel,
        "model_family": config[
            "model_family"
        ],
        "calibration_ensemble_mode": config[
            "calibration_ensemble_mode"
        ],

        "id_column": "Variant_ID",
        "label_column": "Label",

        # Schema expected AFTER only the script's raw normalization.
        "raw_feature_columns": Xraw.columns.tolist(),

        # Self-contained cleaning recipe for future inference.
        "raw_normalization": {
            "mode": "VALIDATED_LEGACY_EXACT",
            "replace_dot_slash_dot_with_nan_all_panels": True,
            "master_only_numeric_prefixes_to_coerce": [
                "AL_",
            ],
            "master_dataframe_name": "genel_df",
            "ek_numeric_conversion_applied": False,
            "global_al_numeric_conversion_applied": False,
            "label_expected_values": [
                0,
                1,
            ],
            "training_normalization_report": raw_normalization_report,
        },

        "use_missing_masks": bool(
            config[
                "use_missing_masks"
            ]
        ),

        "legacy_mask_routing": bool(
            prep_state.get(
                "legacy_mask_routing",
                False,
            )
        ),

        "cat_aa_missing_masks_enter_categorical_branch": bool(
            panel == "CANCER"
            and prep_state.get(
                "legacy_mask_routing",
                False,
            )
        ),

        "cancer_deviation_gate": (
            float(
                CANCER_DEVIATION_GATE
            )
            if panel == "CANCER"
            else None
        ),

        "random_state": RANDOM_STATE,

        "training_rows_real": int(
            len(
                y
            )
        ),
        "training_pathogenic": int(
            (
                y == 1
            ).sum()
        ),
        "training_benign": int(
            (
                y == 0
            ).sum()
        ),
        "training_prior_real": float(
            y.mean()
        ),

        "training_dataframe_fingerprint_after_normalization": dataframe_fingerprint(
            raw_df
        ),

        "preprocessing_state": prep_state,
        "feature_state": feature_state,

        "n_preprocessed_features": int(
            Xpre.shape[1]
        ),
        "n_selected_features": int(
            Xfinal.shape[1]
        ),

        "selected_missing_masks": [
            c
            for c in feature_state[
                "selected_features"
            ]
            if str(c).endswith(
                "_is_missing"
            )
        ],

        "model": model,
        "model_training_info": train_info,

        "positive_class": 1,
        "negative_class": 0,
        "positive_class_name": "Pathogenic",
        "negative_class_name": "Benign",
        "predicted_probability_semantics": "P(Pathogenic=1)",

        "calibration_prior": float(
            calibration_prior
        ),

        "official_test_prior": float(
            config[
                "official_prior"
            ]
        ),

        "prior_policy": config[
            "prior_policy"
        ],

        "threshold_policy": threshold_policy,

        "threshold_optimization_metric": "(F1+MCC)/2",

        # OOF is retained for reproducibility / audit.
        "oof_probability_metrics": oof_result[
            "probability_metrics"
        ],

        "oof_y": oof_result[
            "y"
        ],

        "oof_pathogenic_probability": oof_result[
            "p"
        ],
    }

    return artifact


# =============================================================================
# 14) In-memory artifact health check BEFORE PKL save
#     (NOT the later reload / raw-test / JSON smoke test)
# =============================================================================

def validate_artifact_in_memory(
    artifact,
    normalized_training_df,
):
    Xraw = (
        normalized_training_df.drop(
            columns=[
                "Label",
                "Variant_ID",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    expected_raw = artifact[
        "raw_feature_columns"
    ]

    # Exact training schema should match.
    if list(
        Xraw.columns
    ) != list(
        expected_raw
    ):
        raise RuntimeError(
            f"{artifact['panel']}: raw training feature schema mismatch before save."
        )

    Xpre = transform_preprocessor(
        Xraw,
        artifact[
            "preprocessing_state"
        ],
    )

    Xfinal = transform_feature_stage(
        Xpre,
        artifact[
            "feature_state"
        ],
    )

    if Xfinal.shape[1] != artifact[
        "n_selected_features"
    ]:
        raise RuntimeError(
            f"{artifact['panel']}: selected feature count mismatch before save."
        )

    p = artifact[
        "model"
    ].predict_proba(
        Xfinal
    )[:, 1]

    if len(
        p
    ) != len(
        normalized_training_df
    ):
        raise RuntimeError(
            f"{artifact['panel']}: prediction length mismatch before save."
        )

    if not np.isfinite(
        p
    ).all():
        raise RuntimeError(
            f"{artifact['panel']}: non-finite probability before save."
        )

    if (
        (p < 0).any()
        or (p > 1).any()
    ):
        raise RuntimeError(
            f"{artifact['panel']}: probability outside [0,1] before save."
        )

    policy = artifact[
        "threshold_policy"
    ]

    if policy[
        "policy"
    ] == "FIXED_PRIOR":
        t = float(
            policy[
                "fixed_threshold"
            ]
        )

        if not (
            0 <= t <= 1
        ):
            raise RuntimeError(
                f"{artifact['panel']}: invalid fixed threshold."
            )

    else:
        thresholds = np.asarray(
            list(
                policy[
                    "threshold_map"
                ].values()
            ),
            dtype=float,
        )

        if (
            not np.isfinite(
                thresholds
            ).all()
            or (
                thresholds < 0
            ).any()
            or (
                thresholds > 1
            ).any()
        ):
            raise RuntimeError(
                f"{artifact['panel']}: invalid threshold map."
            )

    return {
        "panel": artifact[
            "panel"
        ],
        "rows_checked": int(
            len(
                p
            )
        ),
        "probability_min": float(
            np.min(
                p
            )
        ),
        "probability_max": float(
            np.max(
                p
            )
        ),
        "probability_mean": float(
            np.mean(
                p
            )
        ),
        "finite_probabilities": True,
        "schema_ok": True,
        "threshold_policy_ok": True,
    }


# =============================================================================
# 15) Atomic PKL save
# =============================================================================

def atomic_joblib_dump(
    obj,
    final_path,
):
    final_path = Path(
        final_path
    )

    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = final_path.with_suffix(
        final_path.suffix
        + ".tmp"
    )

    joblib.dump(
        obj,
        tmp_path,
        compress=3,
    )

    os.replace(
        tmp_path,
        final_path,
    )


def persistent_drive_path(
    local_path,
):
    """
    Map a path inside /content/TEKNOFEST_FINAL_PACKAGE_V2
    to the identical relative path inside:
        /content/drive/MyDrive/TEKNOFEST_FINAL_PACKAGE_V2
    """
    local_path = Path(
        local_path
    )

    relative = local_path.relative_to(
        OUTPUT_ROOT
    )

    return (
        DRIVE_OUTPUT_ROOT
        / relative
    )


def copy_file_to_drive_immediately(
    local_path,
):
    """
    Persist one completed file to Google Drive immediately.
    This is called right after each model PKL is saved.
    """
    local_path = Path(
        local_path
    )

    drive_path = persistent_drive_path(
        local_path
    )

    drive_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_drive_path = drive_path.with_suffix(
        drive_path.suffix
        + ".tmp"
    )

    shutil.copy2(
        local_path,
        tmp_drive_path,
    )

    os.replace(
        tmp_drive_path,
        drive_path,
    )

    return drive_path


def sync_complete_package_to_drive():
    """
    Final full synchronization after all reports/environment files are written.
    Existing Drive files are updated in place.
    """
    if not OUTPUT_ROOT.exists():
        raise RuntimeError(
            "Local package does not exist; cannot sync to Drive."
        )

    shutil.copytree(
        OUTPUT_ROOT,
        DRIVE_OUTPUT_ROOT,
        dirs_exist_ok=True,
    )

    return DRIVE_OUTPUT_ROOT


# =============================================================================
# 16) Environment / reproducibility files
# =============================================================================

def full_environment_freeze_lines():
    rows = []

    try:
        for dist in importlib_metadata.distributions():
            name = (
                dist.metadata.get(
                    "Name"
                )
                or dist.metadata.get(
                    "Summary"
                )
                or "UNKNOWN"
            )

            version = dist.version

            if name != "UNKNOWN":
                rows.append(
                    f"{name}=={version}"
                )
    except Exception:
        return []

    return sorted(
        set(
            rows
        ),
        key=str.lower,
    )


def save_environment_files():
    versions = {
        "created_at_utc": now_utc_iso(),
        "python": sys.version,
        "platform": platform.platform(),

        "numpy": package_version(
            "numpy"
        ),
        "pandas": package_version(
            "pandas"
        ),
        "scikit-learn": package_version(
            "scikit-learn"
        ),
        "scikit-learn_calibrated_classifier_default_ensemble": sklearn_ensemble_default(),
        "imbalanced-learn": package_version(
            "imbalanced-learn"
        ),
        "lightgbm": package_version(
            "lightgbm"
        ),
        "xgboost": package_version(
            "xgboost"
        ),
        "joblib": package_version(
            "joblib"
        ),
    }

    with open(
        ENV_DIR
        / "environment_versions.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            versions,
            f,
            ensure_ascii=False,
            indent=2,
        )

    direct_requirements = [
        f"numpy=={versions['numpy']}",
        f"pandas=={versions['pandas']}",
        f"scikit-learn=={versions['scikit-learn']}",
        f"imbalanced-learn=={versions['imbalanced-learn']}",
        f"lightgbm=={versions['lightgbm']}",
        f"xgboost=={versions['xgboost']}",
        f"joblib=={versions['joblib']}",
    ]

    with open(
        ENV_DIR
        / "requirements_exact.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(
                direct_requirements
            )
            + "\n"
        )

    # Full installed environment snapshot for later offline packaging.
    full_freeze = full_environment_freeze_lines()

    with open(
        ENV_DIR
        / "pip_freeze_full.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(
                full_freeze
            )
            + (
                "\n"
                if full_freeze
                else ""
            )
        )

    return versions


# =============================================================================
# 17) Optional copy of this exact production script into package
# =============================================================================

def copy_this_script_if_possible():
    try:
        if "__file__" in globals():
            src = Path(
                __file__
            )

            if src.exists():
                dst = (
                    OUTPUT_ROOT
                    / src.name
                )

                # Avoid SameFileError.
                if src.resolve() != dst.resolve():
                    shutil.copy2(
                        src,
                        dst,
                    )

                return str(
                    dst
                )
    except Exception:
        pass

    return None


# =============================================================================
# 18) MAIN — OOF -> freeze -> 100% fit -> validate -> PKL save
# =============================================================================

all_fold_reports = []
all_policy_reports = []
normalization_reports = []
artifact_health_reports = []

summary_rows = []
saved_models = {}


print(
    "\n"
    + "#"
    * 170
)

print(
    "TEKNOFEST 2026 — FINAL PRODUCTION FREEZE V2 (LOCKED)"
)

print(
    "LEGACY-EXACT CLEANING -> FINAL OOF -> FREEZE POLICY -> 100% FIT -> HEALTH CHECK -> PKL SAVE"
)

print(
    "Cleaning LOCK: './.' -> NaN ALL panels | AL_* -> numeric MASTER only | EK_* unchanged"
)

print(
    "Official threshold objective = (F1 + MCC) / 2"
)

print(
    "PAH calibration ensemble = DEFAULT/AUTO | CFTR = ensemble=False"
)

print(
    "CANCER lock = legacy CAT/AA mask routing + EM deviation gate ±0.08"
)


print(
    "DUAL SAVE LOCK = every completed model -> /content + Google Drive immediately"
)

print(
    f"Installed sklearn CalibratedClassifierCV default ensemble = {sklearn_ensemble_default()}"
)

print(
    "#"
    * 170
)


for panel, config in PANELS.items():
    print(
        "\n"
        + "="
        * 170
    )

    print(
        panel
    )

    print(
        "="
        * 170
    )

    # -------------------------------------------------------------------------
    # [0/5] Self-contained raw normalization
    # -------------------------------------------------------------------------
    print(
        "\n[0/5] Self-contained raw normalization..."
    )

    df, normalization_report = get_dataframe(
        config[
            "df_name"
        ]
    )

    normalization_report[
        "panel"
    ] = panel

    normalization_reports.append(
        normalization_report
    )

    y = df[
        "Label"
    ].astype(
        int
    )

    print(
        f"Rows={len(df)} | "
        f"P={(y == 1).sum()} | "
        f"B={(y == 0).sum()} | "
        f"train_prior={y.mean():.6f} | "
        f"official_prior={config['official_prior']:.6f}"
    )

    print(
        f"masks={config['use_missing_masks']} | "
        f"model={config['model_family']} | "
        f"policy={config['prior_policy']} | "
        f"calibration_ensemble={config['calibration_ensemble_mode']}"
    )

    print(
        f"'./.' tokens replaced={normalization_report['explicit_dot_slash_dot_tokens_replaced']} | "
        f"MASTER-only AL numeric={normalization_report['master_only_al_numeric_conversion']} | "
        f"EK numeric=False | "
        f"new numeric-coercion NaN={normalization_report['new_nan_from_numeric_coercion_total']}"
    )

    # -------------------------------------------------------------------------
    # [1/5] Final leakage-safe OOF
    # -------------------------------------------------------------------------
    print(
        "\n[1/5] Final leakage-safe OOF..."
    )

    oof = generate_final_oof(
        panel,
        df,
        config[
            "use_missing_masks"
        ],
    )

    probability_report = oof[
        "probability_metrics"
    ]

    print(
        f"PR-AUC(path)={probability_report['pr_auc_pathogenic']:.6f} | "
        f"PR-AUC(benign)={probability_report['pr_auc_benign']:.6f} | "
        f"ROC={probability_report['roc_auc']:.6f} | "
        f"Brier={probability_report['brier']:.6f} | "
        f"LogLoss={probability_report['logloss']:.6f}"
    )

    all_fold_reports.append(
        oof[
            "fold_diagnostics"
        ]
    )

    # -------------------------------------------------------------------------
    # [2/5] Freeze threshold/prior policy
    # -------------------------------------------------------------------------
    print(
        "\n[2/5] Freeze threshold/prior policy using official (F1+MCC)/2..."
    )

    policy = build_threshold_policy(
        panel,
        oof[
            "y"
        ],
        oof[
            "p"
        ],
        config[
            "official_prior"
        ],
    )

    policy_report = evaluate_final_policy(
        panel,
        oof[
            "y"
        ],
        oof[
            "p"
        ],
        oof[
            "calibration_prior"
        ],
        policy,
        config[
            "official_prior"
        ],
    )

    print(
        policy_report.to_string(
            index=False
        )
    )

    all_policy_reports.append(
        policy_report
    )

    official_idx = (
        policy_report[
            "true_prior"
        ]
        - config[
            "official_prior"
        ]
    ).abs().idxmin()

    official_row = policy_report.loc[
        official_idx
    ]

    # -------------------------------------------------------------------------
    # [3/5] Fit complete pipeline on 100% REAL labeled training data
    # -------------------------------------------------------------------------
    print(
        "\n[3/5] Fit preprocessing + Top80 + scaler + FINAL model on 100% real labeled data..."
    )

    artifact = fit_final_100_percent_artifact(
        panel,
        df,
        normalization_report,
        config,
        oof,
        policy,
    )

    # -------------------------------------------------------------------------
    # [4/5] In-memory artifact health check BEFORE save
    # -------------------------------------------------------------------------
    print(
        "\n[4/5] In-memory artifact health check..."
    )

    health = validate_artifact_in_memory(
        artifact,
        df,
    )

    artifact[
        "pre_save_health_check"
    ] = health

    artifact_health_reports.append(
        health
    )

    print(
        "HEALTH OK | "
        f"p-range=[{health['probability_min']:.6f}, {health['probability_max']:.6f}] | "
        f"p-mean={health['probability_mean']:.6f}"
    )

    # -------------------------------------------------------------------------
    # [5/5] Save final PKL atomically
    # -------------------------------------------------------------------------
    print(
        "\n[5/5] Save FINAL PKL..."
    )

    panel_dir = MODELS_DIR / panel

    panel_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pkl_path = (
        panel_dir
        / f"{panel}_FINAL_V2.pkl"
    )

    atomic_joblib_dump(
        artifact,
        pkl_path,
    )

    # CRITICAL CHECKPOINT:
    # persist this completed panel to Google Drive immediately.
    drive_pkl_path = copy_file_to_drive_immediately(
        pkl_path
    )

    saved_models[
        panel
    ] = str(
        pkl_path
    )

    print(
        f"SAVED LOCAL -> {pkl_path}"
    )

    print(
        f"SAVED DRIVE -> {drive_pkl_path}"
    )

    # Decision note.
    if panel == "PAH":
        decision_note = (
            f"fixed_prior={artifact['official_test_prior']:.6f}, "
            f"fixed_threshold={artifact['threshold_policy']['fixed_threshold']:.4f}, "
            f"calibration_ensemble=DEFAULT"
        )
    else:
        fallback_threshold = lookup_threshold(
            artifact[
                "threshold_policy"
            ],
            artifact[
                "official_test_prior"
            ],
        )

        if panel == "CANCER":
            decision_note = (
                f"EM+RangeSafetyNet+DeviationGate±{CANCER_DEVIATION_GATE:.2f}, "
                f"legacy_CAT_AA_mask_routing=True, "
                f"fallback_prior={artifact['official_test_prior']:.6f}, "
                f"fallback_threshold={fallback_threshold:.4f}, "
                f"calibration_ensemble={config['calibration_ensemble_mode']}"
            )
        else:
            decision_note = (
                f"EM+SafetyNet, "
                f"fallback_prior={artifact['official_test_prior']:.6f}, "
                f"fallback_threshold={fallback_threshold:.4f}, "
                f"calibration_ensemble={config['calibration_ensemble_mode']}"
            )

    summary_rows.append({
        "panel": panel,
        "rows": int(
            len(
                df
            )
        ),
        "pathogenic": int(
            (
                y == 1
            ).sum()
        ),
        "benign": int(
            (
                y == 0
            ).sum()
        ),
        "train_prior": float(
            y.mean()
        ),

        "raw_dot_slash_dot_replacements": int(
            normalization_report[
                "explicit_dot_slash_dot_tokens_replaced"
            ]
        ),
        "raw_numeric_coercion_new_nan": int(
            normalization_report[
                "new_nan_from_numeric_coercion_total"
            ]
        ),

        "use_missing_masks": bool(
            config[
                "use_missing_masks"
            ]
        ),

        "legacy_mask_routing": bool(
            artifact.get(
                "legacy_mask_routing",
                False,
            )
        ),

        "deviation_gate": (
            float(
                CANCER_DEVIATION_GATE
            )
            if panel == "CANCER"
            else np.nan
        ),

        "model_family": config[
            "model_family"
        ],

        "calibration_ensemble_mode": config[
            "calibration_ensemble_mode"
        ],

        "prior_policy": config[
            "prior_policy"
        ],

        "official_test_prior": float(
            config[
                "official_prior"
            ]
        ),

        "oof_pr_auc_pathogenic": float(
            probability_report[
                "pr_auc_pathogenic"
            ]
        ),
        "oof_pr_auc_benign": float(
            probability_report[
                "pr_auc_benign"
            ]
        ),
        "oof_roc_auc": float(
            probability_report[
                "roc_auc"
            ]
        ),
        "oof_brier": float(
            probability_report[
                "brier"
            ]
        ),
        "oof_logloss": float(
            probability_report[
                "logloss"
            ]
        ),

        "official_prior_f1": float(
            official_row[
                "production_f1_mean"
            ]
        ),
        "official_prior_mcc": float(
            official_row[
                "production_mcc_mean"
            ]
        ),
        "official_panel_score_f1_mcc": float(
            official_row[
                "production_panel_score_mean"
            ]
        ),

        "mean_pm5_panel_score": float(
            policy_report[
                "production_panel_score_mean"
            ].mean()
        ),

        "worst_pm5_panel_score": float(
            policy_report[
                "production_panel_score_mean"
            ].min()
        ),

        "n_preprocessed_features_final": int(
            artifact[
                "n_preprocessed_features"
            ]
        ),
        "n_selected_features_final": int(
            artifact[
                "n_selected_features"
            ]
        ),
        "n_selected_missing_masks_final": int(
            len(
                artifact[
                    "selected_missing_masks"
                ]
            )
        ),

        "calibration_prior": float(
            artifact[
                "calibration_prior"
            ]
        ),

        "decision_note": decision_note,

        "pre_save_health_ok": True,

        "pkl_path": str(
            pkl_path
        ),

        "drive_pkl_path": str(
            drive_pkl_path
        ),
    })


# =============================================================================
# 19) Save reports
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)

folds_df = pd.concat(
    all_fold_reports,
    ignore_index=True,
)

policy_df = pd.concat(
    all_policy_reports,
    ignore_index=True,
)

normalization_df = pd.DataFrame(
    normalization_reports
)

# Dict column -> JSON string for clean CSV.
if (
    "new_nan_from_numeric_coercion_by_column"
    in normalization_df.columns
):
    normalization_df[
        "new_nan_from_numeric_coercion_by_column"
    ] = normalization_df[
        "new_nan_from_numeric_coercion_by_column"
    ].apply(
        lambda x: json.dumps(
            x,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

health_df = pd.DataFrame(
    artifact_health_reports
)

summary_df.to_csv(
    REPORTS_DIR
    / "FINAL_FREEZE_SUMMARY_V2.csv",
    index=False,
    encoding="utf-8-sig",
)

folds_df.to_csv(
    REPORTS_DIR
    / "FINAL_OOF_FOLD_DIAGNOSTICS_V2.csv",
    index=False,
    encoding="utf-8-sig",
)

policy_df.to_csv(
    REPORTS_DIR
    / "FINAL_STRESS_POLICY_REPORT_V2.csv",
    index=False,
    encoding="utf-8-sig",
)

normalization_df.to_csv(
    REPORTS_DIR
    / "RAW_NORMALIZATION_REPORT_V2.csv",
    index=False,
    encoding="utf-8-sig",
)

health_df.to_csv(
    REPORTS_DIR
    / "PRE_SAVE_ARTIFACT_HEALTH_V2.csv",
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 20) Save environment / package metadata
# =============================================================================

environment_versions = save_environment_files()

copied_script = copy_this_script_if_possible()

metadata = {
    "artifact_version": ARTIFACT_VERSION,
    "created_at_utc": now_utc_iso(),

    "competition": "TEKNOFEST 2026 Sağlıkta Yapay Zekâ",
    "competition_level": "UNIVERSITE_VE_UZERI",

    "positive_class": 1,
    "positive_class_name": "Pathogenic",
    "negative_class": 0,
    "negative_class_name": "Benign",

    "raw_cleaning_is_self_contained": True,
    "raw_cleaning": {
        "mode": "VALIDATED_LEGACY_EXACT",
        "replace_dot_slash_dot_with_nan_all_panels": True,
        "master_only_numeric_prefixes_coerced": [
            "AL_",
        ],
        "master_dataframe_name": "genel_df",
        "ek_numeric_conversion_applied": False,
        "global_al_numeric_conversion_applied": False,
        "exploratory_dataset_list_cell_required": False,
        "exploratory_visualization_cell_required": False,
    },

    "panel_scoring_rule": "(F1 + MCC) / 2",
    "pr_auc_role": "tie-breaker",
    "threshold_selection_metric": "(F1 + MCC) / 2",

    "cancer_final_lock": {
        "legacy_cat_aa_mask_routing": True,
        "range_safety_gate": [
            EM_MIN_PRIOR,
            EM_MAX_PRIOR,
        ],
        "deviation_gate": CANCER_DEVIATION_GATE,
        "deviation_center": PANELS[
            "CANCER"
        ][
            "official_prior"
        ],
        "threshold_objective": "(F1+MCC)/2",
    },

    "frozen_calibration_ensemble_modes": {
        panel: config[
            "calibration_ensemble_mode"
        ]
        for panel, config in PANELS.items()
    },

    "installed_sklearn_calibrated_classifier_default_ensemble": sklearn_ensemble_default(),

    "models": saved_models,

    "environment_versions": environment_versions,

    "production_script_copy": copied_script,
}

with open(
    OUTPUT_ROOT
    / "FINAL_PACKAGE_METADATA_V2.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2,
    )


# =============================================================================
# 20.1) FINAL FULL PACKAGE SYNC TO GOOGLE DRIVE
# =============================================================================

drive_package_path = sync_complete_package_to_drive()

print(
    "\nFINAL PACKAGE SYNCED TO GOOGLE DRIVE ->",
    drive_package_path,
)


# =============================================================================
# 21) Final console summary
# =============================================================================

print(
    "\n"
    + "="
    * 200
)

print(
    "FINAL PRODUCTION TRAINING SUMMARY — V2 LOCKED"
)

print(
    "="
    * 200
)

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    360,
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "="
    * 200
)

print(
    "\nPackage saved locally at:",
    OUTPUT_ROOT.resolve(),
)

print(
    "Package saved persistently in Google Drive at:",
    DRIVE_OUTPUT_ROOT.resolve(),
)

print(
    """
FINAL V2 PACKAGE CONTENTS
-------------------------
Saved in BOTH:
  /content/TEKNOFEST_FINAL_PACKAGE_V2/
  /content/drive/MyDrive/TEKNOFEST_FINAL_PACKAGE_V2/

TEKNOFEST_FINAL_PACKAGE_V2/
  models/
    MASTER/MASTER_FINAL_V2.pkl
    CANCER/CANCER_FINAL_V2.pkl
    PAH/PAH_FINAL_V2.pkl
    CFTR/CFTR_FINAL_V2.pkl

  reports/
    FINAL_FREEZE_SUMMARY_V2.csv
    FINAL_OOF_FOLD_DIAGNOSTICS_V2.csv
    FINAL_STRESS_POLICY_REPORT_V2.csv
    RAW_NORMALIZATION_REPORT_V2.csv
    PRE_SAVE_ARTIFACT_HEALTH_V2.csv

  environment/
    environment_versions.json
    requirements_exact.txt
    pip_freeze_full.txt

  FINAL_PACKAGE_METADATA_V2.json
  [copy of this production script, when run from a .py file]

INTENTIONALLY NOT DONE HERE
---------------------------
1) Reload saved PKLs
2) Pass truly raw Label-less test data
3) Prediction
4) JSON serialization
5) Network-disconnected offline smoke test

Those belong to the NEXT separate inference/smoke-test script.
"""
)
