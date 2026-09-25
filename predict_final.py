# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

# ============================================================
# CONFIG — COMPETITION DAY
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "TEKNOFEST_TEST_DATASETS"
MODELS_DIR = ROOT / "models"

# ============================================================
# OFFICIAL TEAM INFORMATION — DO NOT CHANGE
# ============================================================

TEAM_NAME = "Biyoinformatikçiler"
TEAM_ID = "882005"
APPLICATION_ID = "4878966"
COMPETITION_LEVEL = "UNIVERSITE_VE_UZERI"

# Official recommended output name:
# TEAM_<TAKIM_ID>_FINAL.json

OUTPUT_JSON = ROOT / f"TEAM_{TEAM_ID}_FINAL.json"


# ============================================================
# COMPETITION DAY — EDIT ONLY THESE 4 FILE NAMES
# ============================================================

MASTER_FILE = "YARISMA_TEST_MASTER.csv"
CANCER_FILE = "YARISMA_TEST_KANSER.csv"
PAH_FILE = "YARISMA_TEST_PAH.csv"
CFTR_FILE = "YARISMA_TEST_CFTR.csv"

# ============================================================
# DO NOT EDIT BELOW THIS LINE 
# ============================================================

PANEL_ORDER = [
    "MASTER",
    "CANCER",
    "PAH",
    "CFTR",
]

DATA_PATHS = {
    "MASTER": DATA_DIR / MASTER_FILE,
    "CANCER": DATA_DIR / CANCER_FILE,
    "PAH": DATA_DIR / PAH_FILE,
    "CFTR": DATA_DIR / CFTR_FILE,
}

MODEL_PATHS = {
    "MASTER": MODELS_DIR / "MASTER" / "MASTER_FINAL_V2_PORTABLE.pkl",
    "CANCER": MODELS_DIR / "CANCER" / "CANCER_FINAL_V2.pkl",
    "PAH": MODELS_DIR / "PAH" / "PAH_FINAL_V2.pkl",
    "CFTR": MODELS_DIR / "CFTR" / "CFTR_FINAL_V2.pkl",
}

MASTER_XGB_PATH = (
    MODELS_DIR
    / "MASTER"
    / "MASTER_XGB_V2.ubj"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():

    artifacts = {}

    for panel in PANEL_ORDER:

        path = MODEL_PATHS[panel]

        if not path.exists():
            raise FileNotFoundError(
                f"{panel}: model not found:\n{path}"
            )

        artifacts[panel] = joblib.load(path)

    # --------------------------------------------------------
    # Restore portable XGBoost inside MASTER
    # --------------------------------------------------------

    if not MASTER_XGB_PATH.exists():
        raise FileNotFoundError(
            f"MASTER XGBoost UBJ not found:\n"
            f"{MASTER_XGB_PATH}"
        )

    master = artifacts["MASTER"]

    stack = (
        master["model"]
        .calibrated_classifiers_[0]
        .estimator
        .named_steps["stacking"]
    )

    xgb = XGBClassifier()
    xgb.load_model(MASTER_XGB_PATH)

    portable_info = master.get(
        "portable_xgboost",
        {}
    )

    xgb_index = portable_info.get(
        "stack_estimators_index",
        1
    )

    stack.estimators_[xgb_index] = xgb
    stack.named_estimators_["xgb"] = xgb

    return artifacts


# ============================================================
# RAW NORMALIZATION — EXACT LEGACY V2
# ============================================================

def normalize_raw_test(
    df,
    panel,
    artifact,
):

    df = df.copy()

    if "Label" in df.columns:
        raise RuntimeError(
            f"{panel}: Label must NOT exist in test data."
        )

    id_col = artifact["id_column"]

    if id_col not in df.columns:
        raise RuntimeError(
            f"{panel}: missing {id_col}."
        )

    if df[id_col].isna().any():
        raise RuntimeError(
            f"{panel}: Variant_ID contains missing values."
        )

    if (
        df[id_col]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"{panel}: duplicate Variant_ID detected."
        )

    # './.' -> NaN for ALL panels
    df = df.replace(
        r"^\s*\./\.\s*$",
        np.nan,
        regex=True,
    )

    # MASTER ONLY:
    # AL_* -> numeric
    if panel == "MASTER":

        for col in df.columns:

            if str(col).startswith("AL_"):

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

    expected_features = list(
        artifact["raw_feature_columns"]
    )

    actual_features = [
        c for c in df.columns
        if c != id_col
    ]

    missing = [
        c for c in expected_features
        if c not in actual_features
    ]

    extra = [
        c for c in actual_features
        if c not in expected_features
    ]

    if missing:
        raise RuntimeError(
            f"{panel}: missing raw features:\n"
            f"{missing[:20]}"
        )

    if extra:
        raise RuntimeError(
            f"{panel}: unexpected raw features:\n"
            f"{extra[:20]}"
        )

    ids = (
        df[id_col]
        .astype(str)
        .copy()
    )

    Xraw = (
        df[
            expected_features
        ]
        .copy()
        .reset_index(drop=True)
    )

    ids = ids.reset_index(drop=True)

    return ids, Xraw


# ============================================================
# SAVED PREPROCESSOR TRANSFORM
# ============================================================

def transform_preprocessor(
    X_raw,
    state,
):

    X = X_raw.copy()

    # Restore exact training base schema
    for col in state["kept_base_columns"]:

        if col not in X.columns:
            X[col] = np.nan

    X = X[
        state["kept_base_columns"]
    ].copy()

    # --------------------------------------------------------
    # Missing masks from CURRENT test data
    # --------------------------------------------------------

    if state["add_missing_indicators"]:

        masks = {}

        for col in state["missing_sources"]:

            masks[f"{col}_is_missing"] = (
                X[col]
                .isna()
                .astype(np.int8)
            )

        if masks:

            mask_df = pd.DataFrame(
                masks,
                index=X.index,
            )

            X = pd.concat(
                [X, mask_df],
                axis=1,
            )

    # --------------------------------------------------------
    # CAT_1 frequency map from TRAIN
    # --------------------------------------------------------

    if (
        "cat1_frequency_map" in state
        and "CAT_1" in X.columns
    ):

        X["CAT_1"] = X["CAT_1"].map(
            state["cat1_frequency_map"]
        )

    # --------------------------------------------------------
    # CAT / AA ordinal encoding
    # --------------------------------------------------------

    cat_aa = state.get(
        "cat_aa_columns",
        [],
    )

    if cat_aa:

        for col in cat_aa:

            if col not in X.columns:
                X[col] = np.nan

            X[col] = X[col].astype(object)

        X[cat_aa] = (
            state["ordinal_encoder"]
            .transform(
                X[cat_aa]
            )
        )

    # --------------------------------------------------------
    # KNN branch
    # --------------------------------------------------------

    knn_cols = state.get(
        "knn_columns",
        [],
    )

    if knn_cols:

        Z = (
            state["knn_scaler"]
            .transform(
                X[knn_cols]
            )
        )

        Z = (
            state["knn_imputer"]
            .transform(Z)
        )

        if Z.shape[1] != len(knn_cols):

            raise RuntimeError(
                "KNN transform changed column count."
            )

        X[knn_cols] = (
            state["knn_scaler"]
            .inverse_transform(Z)
        )

        if cat_aa:
            X[cat_aa] = X[cat_aa].round()

    # --------------------------------------------------------
    # MICE branch
    # --------------------------------------------------------

    numeric_cols = state.get(
        "numeric_columns",
        [],
    )

    if numeric_cols:

        Znum = (
            state["mice_imputer"]
            .transform(
                X[numeric_cols]
            )
        )

        if Znum.shape[1] != len(
            numeric_cols
        ):

            raise RuntimeError(
                "MICE transform changed column count."
            )

        X[numeric_cols] = Znum

        X[numeric_cols] = (
            X[numeric_cols]
            .clip(lower=0)
        )

    # --------------------------------------------------------
    # One-hot
    # TRAIN used drop_first=True.
    # Test generates all, then exact reindex.
    # --------------------------------------------------------

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

    for col in X.columns:

        if X[col].dtype == bool:
            X[col] = X[col].astype(
                np.int8
            )

    X = X.astype(float)

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():

        raise RuntimeError(
            "Non-finite value remained after preprocessing."
        )

    return X


# ============================================================
# FEATURE SELECTION + ROBUST SCALER
# ============================================================

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

    continuous = state.get(
        "continuous_features",
        [],
    )

    scaler = state.get(
        "robust_scaler"
    )

    if (
        scaler is not None
        and continuous
    ):

        Xs[continuous] = (
            scaler.transform(
                Xs[continuous]
            )
        )

    if not np.isfinite(
        Xs.to_numpy(dtype=float)
    ).all():

        raise RuntimeError(
            "Non-finite value after feature stage."
        )

    return Xs


# ============================================================
# EM PRIOR ESTIMATION
# ============================================================

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

    for _ in range(max_iter):

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

        if abs(new_p - p) < tol:
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


# ============================================================
# PRIOR POLICY
# ============================================================

def choose_prior(
    artifact,
    test_proba,
):

    policy = artifact[
        "threshold_policy"
    ]

    policy_name = policy["policy"]

    # --------------------------------------------------------
    # PAH
    # --------------------------------------------------------

    if policy_name == "FIXED_PRIOR":

        return {
            "em": None,
            "used_prior": float(
                policy["fixed_prior"]
            ),
            "policy_result": "FIXED_PRIOR",
        }

    # --------------------------------------------------------
    # MASTER / CANCER / CFTR
    # --------------------------------------------------------

    em = estimate_test_prior_em(
        artifact[
            "calibration_prior"
        ],
        test_proba,
    )

    em_min = float(
        policy["em_min_prior"]
    )

    em_max = float(
        policy["em_max_prior"]
    )

    fallback = float(
        policy["fallback_prior"]
    )

    # Global safety range
    if (
        not np.isfinite(em)
        or em < em_min
        or em > em_max
    ):

        return {
            "em": em,
            "used_prior": fallback,
            "policy_result": "FALLBACK_RANGE",
        }

    # --------------------------------------------------------
    # CANCER deviation gate
    # --------------------------------------------------------

    if (
        policy_name
        == "EM_SAFETYNET_DEVIATION_GATE"
    ):

        lower = float(
            policy["deviation_lower"]
        )

        upper = float(
            policy["deviation_upper"]
        )

        chosen = float(
            np.clip(
                em,
                lower,
                upper,
            )
        )

        if abs(chosen - em) > 1e-12:

            result = "EM_CLIPPED_DEVIATION"

        else:

            result = "EM_ACCEPTED"

        return {
            "em": em,
            "used_prior": chosen,
            "policy_result": result,
        }

    # MASTER / CFTR accepted EM
    return {
        "em": em,
        "used_prior": float(em),
        "policy_result": "EM_ACCEPTED",
    }


# ============================================================
# THRESHOLD LOOKUP
# ============================================================

def lookup_threshold(
    artifact,
    chosen_prior,
):

    policy = artifact[
        "threshold_policy"
    ]

    if policy["policy"] == "FIXED_PRIOR":

        return float(
            policy["fixed_threshold"]
        )

    grid = np.asarray(
        policy[
            "threshold_prior_grid"
        ],
        dtype=float,
    )

    nearest = float(
        grid[
            np.argmin(
                np.abs(
                    grid - chosen_prior
                )
            )
        ]
    )

    key = f"{nearest:.6f}"

    return float(
        policy[
            "threshold_map"
        ][key]
    )


# ============================================================
# PANEL INFERENCE
# ============================================================

def predict_panel(
    panel,
    raw_df,
    artifact,
):

    ids, Xraw = normalize_raw_test(
        raw_df,
        panel,
        artifact,
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

    proba = (
        artifact["model"]
        .predict_proba(Xfinal)[:, 1]
    )

    proba = np.asarray(
        proba,
        dtype=float,
    )

    if not np.isfinite(proba).all():
        raise RuntimeError(
            f"{panel}: probability contains NaN/Infinity."
        )

    if (
        (proba < 0).any()
        or (proba > 1).any()
    ):
        raise RuntimeError(
            f"{panel}: probability outside [0,1]."
        )

    prior_info = choose_prior(
        artifact,
        proba,
    )

    threshold = lookup_threshold(
        artifact,
        prior_info[
            "used_prior"
        ],
    )

    pred = (
        proba >= threshold
    ).astype(int)

    result = pd.DataFrame({
        "id": ids,
        "panel": panel,
        "predicted_class": pred,
        "predicted_prob": proba,
    })

    return (
        result,
        prior_info,
        threshold,
    )


# ============================================================
# JSON VALIDATION
# ============================================================

def validate_predictions(
    predictions,
):

    required = {
        "id",
        "panel",
        "predicted_class",
        "predicted_prob",
    }

    if set(predictions.columns) != required:
        raise RuntimeError(
            "Prediction columns are invalid."
        )

    if predictions["id"].isna().any():
        raise RuntimeError(
            "Missing prediction ID."
        )

    if (
        predictions["id"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Empty prediction ID."
        )

    valid_panels = set(
        PANEL_ORDER
    )

    if not set(
        predictions["panel"].unique()
    ).issubset(valid_panels):

        raise RuntimeError(
            "Invalid panel name."
        )

    if not set(
        predictions[
            "predicted_class"
        ].unique()
    ).issubset({0, 1}):

        raise RuntimeError(
            "Invalid predicted_class."
        )

    p = predictions[
        "predicted_prob"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(p).all():
        raise RuntimeError(
            "NaN/Infinity in predicted_prob."
        )

    if (
        (p < 0).any()
        or (p > 1).any()
    ):
        raise RuntimeError(
            "predicted_prob outside [0,1]."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("TEKNOFEST 2026 — OFFLINE WINDOWS TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading models...")

    artifacts = load_artifacts()

    print("All 4 models loaded successfully.")

    all_predictions = []

    # --------------------------------------------------------
    # Predict each panel
    # --------------------------------------------------------

    print("\nRunning inference...\n")

    for panel in PANEL_ORDER:

        csv_path = DATA_PATHS[panel]

        if not csv_path.exists():
            raise FileNotFoundError(
                f"{panel}: CSV not found:\n"
                f"{csv_path}"
            )

        raw_df = pd.read_csv(
            csv_path
        )

        result, prior_info, threshold = (
            predict_panel(
                panel,
                raw_df,
                artifacts[panel],
            )
        )

        all_predictions.append(
            result
        )

        em_text = (
            "N/A"
            if prior_info["em"] is None
            else f"{prior_info['em']:.6f}"
        )

        class0 = int(
            (
                result[
                    "predicted_class"
                ] == 0
            ).sum()
        )

        class1 = int(
            (
                result[
                    "predicted_class"
                ] == 1
            ).sum()
        )

        print(
            f"{panel:7s} | "
            f"N={len(result):4d} | "
            f"EM={em_text} | "
            f"used_prior="
            f"{prior_info['used_prior']:.6f} | "
            f"threshold={threshold:.4f} | "
            f"class0={class0:4d} | "
            f"class1={class1:4d} | "
            f"policy="
            f"{prior_info['policy_result']}"
        )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_predictions(
        predictions_df
    )

    # Ensure every input row produced one prediction
    expected_total = sum(
        len(
            pd.read_csv(
                DATA_PATHS[p],
                usecols=["Variant_ID"],
            )
        )
        for p in PANEL_ORDER
    )

    if len(predictions_df) != expected_total:
        raise RuntimeError(
            "Prediction count does not match input rows."
        )

    # --------------------------------------------------------
    # Build official-format JSON
    # --------------------------------------------------------

    json_predictions = []

    for row in predictions_df.itertuples(
        index=False
    ):

        json_predictions.append({
            "id": str(row.id),
            "panel": str(row.panel),
            "predicted_class": str(
                int(row.predicted_class)
            ),
            "predicted_prob": float(
                row.predicted_prob
            ),
        })

    payload = {
        "team_name": TEAM_NAME,
        "team_id": TEAM_ID,
        "application_id": APPLICATION_ID,
        "competition_level": COMPETITION_LEVEL,
        "predictions": json_predictions,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("\n" + "=" * 72)
    print("SUCCESS")
    print("=" * 72)

    print(
        "Total predictions:",
        len(json_predictions)
    )

    print(
        "JSON:",
        OUTPUT_JSON
    )

    print(
        "File size:",
        round(
            OUTPUT_JSON.stat().st_size
            / 1024,
            2,
        ),
        "KB"
    )


if __name__ == "__main__":
    main()