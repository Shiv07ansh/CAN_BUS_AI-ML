
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "prepared" / "set_01" / "train_01"
OUTPUT_DIR = (
    ROOT / "reports" / "set_01" / "stage_6"
    / "feature_subset_experiments"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "attack"
RANDOM_STATE = 42
VALIDATION_SAMPLE_SIZE = 50_000

TRAIN_FILES = [
    "accessory-2.csv",
    "attack-free-1.csv",
    "attack-free-2.csv",
    "DoS-2.csv",
    "force-neutral-2.csv",
    "rpm-1.csv",
    "standstill-1.csv",
]

VALIDATION_FILES = [
    "accessory-1.csv",
    "DoS-1.csv",
    "force-neutral-1.csv",
    "rpm-2.csv",
    "standstill-2.csv",
]

ALL_FEATURES = [
    "arbitration_id",
    "payload_length",
    "byte_0",
    "byte_1",
    "byte_2",
    "byte_3",
    "byte_4",
    "byte_5",
    "byte_6",
    "byte_7",
]

FEATURE_SETS = {
    "F0_original_10": ALL_FEATURES,

    "F1_identifier_only": [
        "arbitration_id",
    ],

    "F2_statistical_shortlist": [
        "arbitration_id",
        "byte_1",
        "byte_2",
        "byte_3",
        "byte_6",
    ],

    "F3_tuned_rf_shortlist": [
        "arbitration_id",
        "byte_0",
        "byte_4",
        "byte_5",
        "byte_6",
    ],

    "F4_payload_only": [
        "payload_length",
        "byte_0",
        "byte_1",
        "byte_2",
        "byte_3",
        "byte_4",
        "byte_5",
        "byte_6",
        "byte_7",
    ],

    "F5_combined_shortlist": [
        "arbitration_id",
        "byte_0",
        "byte_1",
        "byte_3",
        "byte_4",
        "byte_5",
        "byte_6",
    ],
}


# Fixed configuration across every feature set.
# This is the baseline RF configuration, not a newly tuned model.
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# DATA LOADING
# ============================================================

def load_files(file_names, description):
    frames = []

    required_columns = ALL_FEATURES + [TARGET]

    for file_name in file_names:
        path = DATA_DIR / file_name

        if not path.exists():
            raise FileNotFoundError(
                f"{description} file not found: {path}"
            )

        print(f"Loading {description}: {file_name}")

        frame = pd.read_csv(path, usecols=required_columns)

        if frame[required_columns].isnull().any().any():
            raise ValueError(
                f"Missing values detected in {file_name}"
            )

        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)

    result[TARGET] = result[TARGET].astype(np.int8)

    return result


def prepare_data():
    print("\nLoading training recordings...")
    train_df = load_files(TRAIN_FILES, "training")

    print("\nLoading validation recordings...")
    val_df = load_files(VALIDATION_FILES, "validation")

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]

    X_val = val_df[ALL_FEATURES]
    y_val = val_df[TARGET]

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError("Both partitions must contain both classes.")

    print("\nCreating fixed validation sample...")

    X_val_sample, _, y_val_sample, _ = train_test_split(
        X_val,
        y_val,
        train_size=VALIDATION_SAMPLE_SIZE,
        stratify=y_val,
        random_state=RANDOM_STATE,
    )

    X_val_sample = X_val_sample.reset_index(drop=True)
    y_val_sample = y_val_sample.reset_index(drop=True)

    print("\nDataset summary")
    print("------------------------------")
    print(f"Training rows: {len(X_train):,}")
    print(f"Training attacks: {int(y_train.sum()):,}")
    print(f"Validation rows available: {len(X_val):,}")
    print(f"Validation sample: {len(X_val_sample):,}")
    print(f"Validation attacks: {int(y_val_sample.sum()):,}")
    print(f"Validation attack prevalence: {y_val_sample.mean():.6%}")

    return X_train, y_train, X_val_sample, y_val_sample


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_model(y_true, probabilities):
    pr_auc = average_precision_score(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)

    # Standard threshold of 0.5
    default_metrics = calculate_metrics(
        y_true, probabilities, 0.5
    )

    # Exploratory threshold optimisation on this validation sample
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        best_threshold = 0.5
    else:
        f1_values = (
            2 * precisions[:-1] * recalls[:-1]
            / (precisions[:-1] + recalls[:-1] + 1e-12)
        )

        best_index = int(np.argmax(f1_values))
        best_threshold = float(thresholds[best_index])

    optimised_metrics = calculate_metrics(
        y_true, probabilities, best_threshold
    )

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "default": default_metrics,
        "optimised": optimised_metrics,
    }


# ============================================================
# EXPERIMENT
# ============================================================

def main():
    X_train, y_train, X_val, y_val = prepare_data()

    results = []

    print("\nStarting feature-subset experiments")
    print("====================================")

    for set_name, features in FEATURE_SETS.items():

        print(f"\n\nExperiment: {set_name}")
        print(f"Features ({len(features)}): {features}")

        model = RandomForestClassifier(**RF_PARAMS)

        start_time = time.time()

        model.fit(
            X_train[features],
            y_train,
        )

        training_seconds = time.time() - start_time

        print(
            f"Training completed in "
            f"{training_seconds / 60:.2f} minutes"
        )

        probabilities = model.predict_proba(
            X_val[features]
        )[:, 1]

        metrics = evaluate_model(y_val, probabilities)

        default = metrics["default"]
        optimised = metrics["optimised"]

        result = {
            "feature_set": set_name,
            "feature_count": len(features),
            "features": ", ".join(features),
            "training_seconds": training_seconds,

            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],

            "default_precision": default["precision"],
            "default_recall": default["recall"],
            "default_f1": default["f1"],
            "default_fpr": default["fpr"],

            "best_threshold": optimised["threshold"],
            "optimised_precision": optimised["precision"],
            "optimised_recall": optimised["recall"],
            "optimised_f1": optimised["f1"],
            "optimised_fpr": optimised["fpr"],
        }

        results.append(result)

        print("\nResults")
        print("------------------------------")
        print(f"PR-AUC: {metrics['pr_auc']:.6f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
        print(f"Default F1: {default['f1']:.6f}")
        print(f"Optimised F1: {optimised['f1']:.6f}")
        print(
            f"Optimised threshold: "
            f"{optimised['threshold']:.6f}"
        )
        print(f"Optimised FPR: {optimised['fpr']:.6f}")

        del model
        del probabilities

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False,
    )

    csv_path = OUTPUT_DIR / "feature_subset_results.csv"
    json_path = OUTPUT_DIR / "experiment_metadata.json"

    results_df.to_csv(csv_path, index=False)

    metadata = {
        "experiment": "Controlled Random Forest feature-subset comparison",
        "random_state": RANDOM_STATE,
        "validation_sample_size": VALIDATION_SAMPLE_SIZE,
        "training_rows": int(len(X_train)),
        "training_attacks": int(y_train.sum()),
        "validation_sample_rows": int(len(X_val)),
        "validation_sample_attacks": int(y_val.sum()),
        "rf_parameters": RF_PARAMS,
        "feature_sets": FEATURE_SETS,
        "training_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "threshold_note": (
            "Optimised thresholds are exploratory. "
            "They were selected and evaluated on the same "
            "fixed validation sample."
        ),
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n\nEXPERIMENT COMPLETE")
    print("====================================")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {csv_path}")
    print(f"Metadata saved to: {json_path}")


if __name__ == "__main__":
    main()
