
from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "prepared" / "set_01" / "train_01"
)

OUTPUT_DIR = (
    ROOT / "reports" / "set_01" / "stage_6"
    / "can_id_controlled_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SAMPLE_SIZE = 50_000
UNSEEN_ID = 0
THRESHOLD = 0.5

TARGET = "attack"
ID_COLUMN = "arbitration_id"

PAYLOAD_FEATURES = [
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

REQUIRED_COLUMNS = [
    ID_COLUMN,
    *PAYLOAD_FEATURES,
    TARGET,
]

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

# Identical number of candidate features at every split.
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "max_features": 3,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

EXPERIMENTS = {
    "E0_integer": "integer",
    "E1_binary": "binary",
    "E2_onehot": "onehot",
    "E3_payload_only": "payload",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_files(file_names, description):
    frames = []

    for filename in file_names:
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        print(f"Loading {description}: {filename}")

        frames.append(
            pd.read_csv(
                path,
                usecols=REQUIRED_COLUMNS,
            )
        )

    return pd.concat(frames, ignore_index=True)


# ============================================================
# VALIDATION
# ============================================================

def validate_data(train_df, val_df, unseen_df):

    for name, frame in [
        ("training", train_df),
        ("validation", val_df),
        ("unseen-ID evaluation", unseen_df),
    ]:
        if frame[REQUIRED_COLUMNS].isnull().any().any():
            raise ValueError(f"Missing values in {name} data.")

        if not frame[TARGET].isin([0, 1]).all():
            raise ValueError(f"Invalid labels in {name} data.")

        ids = frame[ID_COLUMN]

        if not ((ids >= 0) & (ids <= 2047)).all():
            raise ValueError(f"Invalid CAN IDs in {name} data.")

    if (train_df[ID_COLUMN] == UNSEEN_ID).any():
        raise ValueError("ID 0 unexpectedly appears in training.")

    if not (unseen_df[ID_COLUMN] == UNSEEN_ID).all():
        raise ValueError("Unexpected ID in unseen-ID evaluation.")

    if not (unseen_df[TARGET] == 1).all():
        raise ValueError("Unseen-ID records are not all attacks.")

    print("\nInput validation passed.")


# ============================================================
# TRAINING-ONLY ID VOCABULARY
# ============================================================

def fit_categories(train_df):
    return np.sort(
        train_df[ID_COLUMN].unique().astype(np.uint16)
    )


# ============================================================
# FEATURE MATRIX CONSTRUCTION
# ============================================================

def build_matrix(frame, representation, categories):

    ids = frame[ID_COLUMN].to_numpy(dtype=np.uint16)

    payload = frame[
        PAYLOAD_FEATURES
    ].to_numpy(dtype=np.uint16)

    if representation == "integer":

        id_features = ids.reshape(-1, 1)

    elif representation == "binary":

        id_features = np.column_stack([
            ((ids >> bit) & 1).astype(np.uint8)
            for bit in range(10, -1, -1)
        ])

    elif representation == "onehot":

        id_features = np.zeros(
            (len(ids), len(categories)),
            dtype=np.uint8,
        )

        positions = np.searchsorted(categories, ids)

        known = (
            (positions < len(categories))
            & (
                categories[
                    np.minimum(
                        positions,
                        len(categories) - 1,
                    )
                ] == ids
            )
        )

        rows = np.flatnonzero(known)

        id_features[
            rows,
            positions[known],
        ] = 1

    elif representation == "payload":

        return payload

    else:
        raise ValueError(representation)

    return np.column_stack(
        [id_features, payload]
    ).astype(np.uint16, copy=False)


# ============================================================
# METRICS
# ============================================================

def classification_metrics(y_true, probabilities):

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "fpr": float(
            fp / (fp + tn) if fp + tn else 0
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def full_metrics(y_true, probabilities):

    result = classification_metrics(
        y_true,
        probabilities,
    )

    result["pr_auc"] = float(
        average_precision_score(
            y_true,
            probabilities,
        )
    )

    result["roc_auc"] = float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )

    return result


def unseen_metrics(probabilities):

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.uint8)

    total = len(predictions)
    detected = int(predictions.sum())

    return {
        "total": int(total),
        "detected": detected,
        "missed": int(total - detected),
        "recall": float(detected / total) if total else 0.0,
        "mean_probability": float(probabilities.mean()),
        "minimum_probability": float(probabilities.min()),
        "maximum_probability": float(probabilities.max()),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nCONTROLLED CAN ID REPRESENTATION COMPARISON")
    print("============================================")

    train_df = load_files(
        TRAIN_FILES,
        "training",
    )

    val_df = load_files(
        VALIDATION_FILES,
        "validation",
    )

    # Load all ID-0 records from the known validation recording.
    unseen_df = pd.read_csv(
        DATA_DIR / "DoS-1.csv",
        usecols=REQUIRED_COLUMNS,
    )

    unseen_df = unseen_df[
        unseen_df[ID_COLUMN] == UNSEEN_ID
    ].reset_index(drop=True)

    validate_data(
        train_df,
        val_df,
        unseen_df,
    )

    # Reproduce the fixed validation sample.
    val_sample, _, y_sample, _ = train_test_split(
        val_df,
        val_df[TARGET],
        train_size=VALIDATION_SAMPLE_SIZE,
        stratify=val_df[TARGET],
        random_state=RANDOM_STATE,
    )

    val_sample = val_sample.reset_index(drop=True)
    y_sample = y_sample.reset_index(drop=True)

    y_train = train_df[TARGET].to_numpy(dtype=np.uint8)
    y_val = y_sample.to_numpy(dtype=np.uint8)

    categories = fit_categories(train_df)

    unseen_sample_count = int(
        (val_sample[ID_COLUMN] == UNSEEN_ID).sum()
    )

    print("\nDataset summary")
    print("--------------------------------------------")
    print(f"Training rows: {len(train_df):,}")
    print(f"Validation sample: {len(val_sample):,}")
    print(f"Validation attacks: {int(y_val.sum()):,}")
    print(f"Training ID categories: {len(categories)}")
    print(f"Unseen ID: {UNSEEN_ID:#05x}")
    print(f"Unseen records in sample: {unseen_sample_count:,}")
    print(f"All unseen-ID attacks: {len(unseen_df):,}")
    print(f"Fixed max_features: {RF_PARAMS['max_features']}")

    results = []

    for experiment_name, representation in EXPERIMENTS.items():

        print("\n\n============================================")
        print(f"EXPERIMENT: {experiment_name}")
        print("============================================")

        X_train = build_matrix(
            train_df,
            representation,
            categories,
        )

        X_val = build_matrix(
            val_sample,
            representation,
            categories,
        )

        X_unseen = build_matrix(
            unseen_df,
            representation,
            categories,
        )

        print(f"Training matrix: {X_train.shape}")
        print(f"Validation matrix: {X_val.shape}")
        print(f"Unseen matrix: {X_unseen.shape}")

        model = RandomForestClassifier(**RF_PARAMS)

        start = time.time()

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = time.time() - start

        print(
            f"Training time: {training_seconds:.2f} seconds"
        )

        del X_train
        gc.collect()

        # Overall validation
        val_probabilities = model.predict_proba(
            X_val
        )[:, 1]

        overall = full_metrics(
            y_val,
            val_probabilities,
        )

        # Known-ID validation
        known_mask = (
            val_sample[ID_COLUMN].to_numpy() != UNSEEN_ID
        )

        known = full_metrics(
            y_val[known_mask],
            val_probabilities[known_mask],
        )

        # All unseen-ID attack records
        unseen_probabilities = model.predict_proba(
            X_unseen
        )[:, 1]

        unseen = unseen_metrics(
            unseen_probabilities
        )

        result = {
            "experiment": experiment_name,
            "representation": representation,
            "feature_count": int(X_val.shape[1]),
            "training_seconds": float(training_seconds),
            "overall": overall,
            "known_id": known,
            "unseen_id": unseen,
        }

        # ----------------------------------------------------
        # One-hot fallback policy
        # ----------------------------------------------------

        if representation == "onehot":

            # A policy rule, NOT a learned RF prediction.
            # Flag all identifiers absent from training.
            unseen_mask = (
                val_sample[ID_COLUMN].to_numpy()
                == UNSEEN_ID
            )

            fallback_probabilities = (
                val_probabilities.copy()
            )

            fallback_probabilities[unseen_mask] = 1.0

            result["unknown_id_fallback_policy"] = {
                "description": (
                    "Assign probability 1.0 to validation "
                    "records whose ID is absent from training."
                ),
                "overall": full_metrics(
                    y_val,
                    fallback_probabilities,
                ),
                "unseen_id": {
                    "total": int(unseen_mask.sum()),
                    "detected": int(unseen_mask.sum()),
                    "missed": 0,
                    "recall": 1.0,
                },
            }

            print("\nOne-hot + unseen-ID fallback")
            print("--------------------------------------------")
            print(
                "Overall F1:",
                result["unknown_id_fallback_policy"]
                ["overall"]["f1"],
            )
            print(
                "Overall PR-AUC:",
                result["unknown_id_fallback_policy"]
                ["overall"]["pr_auc"],
            )

        print("\nOVERALL VALIDATION")
        print("--------------------------------------------")
        print(f"PR-AUC: {overall['pr_auc']:.6f}")
        print(f"ROC-AUC: {overall['roc_auc']:.6f}")
        print(f"Precision: {overall['precision']:.6f}")
        print(f"Recall: {overall['recall']:.6f}")
        print(f"F1: {overall['f1']:.6f}")
        print(f"FPR: {overall['fpr']:.6f}")

        print("\nKNOWN-ID VALIDATION")
        print("--------------------------------------------")
        print(f"PR-AUC: {known['pr_auc']:.6f}")
        print(f"Recall: {known['recall']:.6f}")
        print(f"F1: {known['f1']:.6f}")

        print("\nUNSEEN-ID ATTACK EVALUATION")
        print("--------------------------------------------")
        print(f"Detected: {unseen['detected']:,}")
        print(f"Missed: {unseen['missed']:,}")
        print(f"Recall: {unseen['recall']:.6f}")

        results.append(result)

        del X_val
        del X_unseen
        del model
        del val_probabilities
        del unseen_probabilities
        gc.collect()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    json_path = (
        OUTPUT_DIR / "controlled_comparison.json"
    )

    csv_rows = []

    for result in results:

        overall = result["overall"]
        known = result["known_id"]
        unseen = result["unseen_id"]

        row = {
            "experiment": result["experiment"],
            "representation": result["representation"],
            "feature_count": result["feature_count"],
            "training_seconds": result["training_seconds"],

            "overall_pr_auc": overall["pr_auc"],
            "overall_roc_auc": overall["roc_auc"],
            "overall_precision": overall["precision"],
            "overall_recall": overall["recall"],
            "overall_f1": overall["f1"],
            "overall_fpr": overall["fpr"],

            "known_id_pr_auc": known["pr_auc"],
            "known_id_recall": known["recall"],
            "known_id_f1": known["f1"],

            "unseen_id_detected": unseen["detected"],
            "unseen_id_missed": unseen["missed"],
            "unseen_id_recall": unseen["recall"],
        }

        if "unknown_id_fallback_policy" in result:

            fallback = result["unknown_id_fallback_policy"]

            row["fallback_overall_pr_auc"] = (
                fallback["overall"]["pr_auc"]
            )

            row["fallback_overall_f1"] = (
                fallback["overall"]["f1"]
            )

            row["fallback_overall_recall"] = (
                fallback["overall"]["recall"]
            )

        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(
        OUTPUT_DIR / "controlled_comparison.csv",
        index=False,
    )

    metadata = {
        "experiment": "Controlled CAN ID representation comparison",
        "random_state": RANDOM_STATE,
        "validation_sample_size": VALIDATION_SAMPLE_SIZE,
        "threshold": THRESHOLD,
        "training_rows": len(train_df),
        "training_attacks": int(y_train.sum()),
        "validation_rows": len(val_df),
        "validation_sample_attacks": int(y_val.sum()),
        "unseen_id": int(UNSEEN_ID),
        "unseen_id_total_records": len(unseen_df),
        "unseen_id_records_in_sample": unseen_sample_count,
        "training_categories": [
            int(x) for x in categories
        ],
        "training_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "rf_parameters": RF_PARAMS,
        "experiments": results,
        "methodology_notes": [
            "All models use max_features=3.",
            "All models use identical training and validation records.",
            "One-hot categories are learned from training data only.",
            "The unseen-ID fallback is a separate rule, not a learned model.",
            "No threshold optimisation is performed.",
        ],
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n\nCONTROLLED EXPERIMENT COMPLETE")
    print("============================================")
    print(f"CSV: {OUTPUT_DIR / 'controlled_comparison.csv'}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
