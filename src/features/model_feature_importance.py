
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "prepared" / "set_01"
MODEL_DIR = ROOT / "models" / "set_01"

OUTPUT_DIR = (
    ROOT / "reports" / "set_01" / "stage_6"
    / "model_feature_importance"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FILES = [
    "accessory-1.csv",
    "DoS-1.csv",
    "force-neutral-1.csv",
    "rpm-2.csv",
    "standstill-2.csv",
]

FEATURES = [
    "arbitration_id",
    "payload_length",
    *[f"byte_{i}" for i in range(8)],
]

TARGET = "attack"

SAMPLE_SIZE = 50_000
RANDOM_SEED = 42
N_REPEATS = 5


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data():

    frames = []

    for filename in VALIDATION_FILES:

        matches = list(DATA_DIR.rglob(filename))

        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {filename}; "
                f"found {len(matches)}"
            )

        filepath = matches[0]

        print(f"Loading validation file: {filename}")

        df = pd.read_csv(filepath)

        df = df[FEATURES + [TARGET]].apply(
            pd.to_numeric,
            errors="raise"
        )

        if df.isna().any().any():
            raise ValueError(
                f"Missing values found in {filename}"
            )

        if not df[TARGET].isin([0, 1]).all():
            raise ValueError(
                f"Invalid labels found in {filename}"
            )

        frames.append(df)

    validation = pd.concat(
        frames,
        ignore_index=True
    )

    return validation


# ============================================================
# LOAD MODEL BUNDLE
# ============================================================

def load_model_bundle(model_name, model_path):

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise TypeError(
            f"{model_name}: Expected dictionary bundle, "
            f"got {type(bundle)}"
        )

    if "model" not in bundle:
        raise KeyError(
            f"{model_name}: 'model' key missing. "
            f"Available keys: {list(bundle.keys())}"
        )

    model = bundle["model"]

    saved_features = bundle.get("features")

    if saved_features is not None:

        if list(saved_features) != FEATURES:
            raise ValueError(
                f"Feature mismatch for {model_name}.\n"
                f"Saved: {saved_features}\n"
                f"Expected: {FEATURES}"
            )

    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"{model_name}: Extracted object "
            f"{type(model)} has no feature_importances_."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{model_name}: Model does not support predict_proba."
        )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 65)

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    validation = load_validation_data()

    X = validation[FEATURES]
    y = validation[TARGET].astype(int)

    print("\nFull validation records:", f"{len(validation):,}")
    print("Full validation attacks:", f"{int(y.sum()):,}")
    print("Full validation prevalence:", f"{y.mean():.6%}")

    if y.nunique() != 2:
        raise ValueError(
            "Validation must contain both classes."
        )

    if SAMPLE_SIZE > len(validation):
        raise ValueError(
            f"Requested sample size {SAMPLE_SIZE:,} "
            f"exceeds validation size {len(validation):,}."
        )

    # --------------------------------------------------------
    # Create fixed stratified validation sample
    # --------------------------------------------------------

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    X_sample = X_sample.reset_index(drop=True)
    y_sample = y_sample.reset_index(drop=True)

    print("\nImportance analysis sample:", f"{len(X_sample):,}")
    print("Sample attacks:", int(y_sample.sum()))
    print("Sample prevalence:", f"{y_sample.mean():.6%}")

    # --------------------------------------------------------
    # Model paths
    # --------------------------------------------------------

    model_paths = {
        "Random Forest Baseline":
            MODEL_DIR / "baseline" / "random_forest.joblib",

        "Random Forest Tuned":
            MODEL_DIR / "tuned" / "random_forest_tuned.joblib",
    }

    summary_rows = []
    permutation_rows = []

    # --------------------------------------------------------
    # Process each model
    # --------------------------------------------------------

    for model_name, model_path in model_paths.items():

        print("\n" + "-" * 65)
        print(model_name)
        print("-" * 65)

        model = load_model_bundle(
            model_name,
            model_path
        )

        print("Loaded estimator:", type(model).__name__)

        # ----------------------------------------------------
        # 1. Impurity-based importance
        # ----------------------------------------------------

        importances = model.feature_importances_

        if len(importances) != len(FEATURES):
            raise ValueError(
                f"{model_name}: Expected {len(FEATURES)} "
                f"feature importances, got {len(importances)}."
            )

        for feature, importance in zip(
            FEATURES,
            importances
        ):

            summary_rows.append({
                "model": model_name,
                "feature": feature,
                "impurity_importance": float(importance),
            })

        # ----------------------------------------------------
        # 2. Validation PR-AUC
        # ----------------------------------------------------

        scores = model.predict_proba(X_sample)[:, 1]

        baseline_ap = average_precision_score(
            y_sample,
            scores
        )

        print(
            f"Validation PR-AUC: {baseline_ap:.6f}"
        )

        # ----------------------------------------------------
        # 3. Permutation importance
        # ----------------------------------------------------

        print(
            f"Calculating permutation importance "
            f"({N_REPEATS} repeats)..."
        )

        result = permutation_importance(
            model,
            X_sample,
            y_sample,
            scoring="average_precision",
            n_repeats=N_REPEATS,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        for i, feature in enumerate(FEATURES):

            permutation_rows.append({
                "model": model_name,
                "feature": feature,
                "permutation_mean": float(
                    result.importances_mean[i]
                ),
                "permutation_std": float(
                    result.importances_std[i]
                ),
            })

        print("Permutation analysis complete.")

    # ========================================================
    # SAVE IMPURITY IMPORTANCE
    # ========================================================

    impurity_df = pd.DataFrame(summary_rows)

    impurity_df["importance_rank"] = (
        impurity_df.groupby("model")["impurity_importance"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    impurity_df = impurity_df.sort_values(
        ["model", "importance_rank"]
    )

    impurity_df.to_csv(
        OUTPUT_DIR / "01_impurity_importance.csv",
        index=False
    )

    # ========================================================
    # SAVE PERMUTATION IMPORTANCE
    # ========================================================

    permutation_df = pd.DataFrame(permutation_rows)

    permutation_df["permutation_rank"] = (
        permutation_df.groupby("model")["permutation_mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    permutation_df = permutation_df.sort_values(
        ["model", "permutation_rank"]
    )

    permutation_df.to_csv(
        OUTPUT_DIR / "02_permutation_importance.csv",
        index=False
    )

    # ========================================================
    # SAVE EXPERIMENT METADATA
    # ========================================================

    metadata = {
        "validation_files": VALIDATION_FILES,
        "full_validation_rows": int(len(validation)),
        "validation_attacks": int(y.sum()),
        "analysis_sample_size": int(len(X_sample)),
        "analysis_sample_attacks": int(y_sample.sum()),
        "random_seed": RANDOM_SEED,
        "permutation_repeats": N_REPEATS,
        "scoring_metric": "average_precision",
        "models": list(model_paths.keys()),
        "features": FEATURES,
        "official_test_partitions_used": False,
    }

    with open(
        OUTPUT_DIR / "03_experiment_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(metadata, f, indent=4)

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("\n" + "=" * 65)
    print("IMPURITY IMPORTANCE")
    print("=" * 65)

    print(
        impurity_df.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("PERMUTATION IMPORTANCE")
    print("=" * 65)

    print(
        permutation_df.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
