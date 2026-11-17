
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "data" / "engineered" / "set_01" / "train_01"
)

REPORT_DIR = (
    ROOT / "reports" / "set_01" / "stage_7"
    / "full_validation_diagnostics"
)

MODEL_DIR = ROOT / "models" / "set_01" / "stage_7"

RANDOM_STATE = 42
TARGET = "attack"

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

BYTE_COLUMNS = [f"byte_{i}" for i in range(8)]

ORIGINAL_FEATURES = [
    "arbitration_id",
    "payload_length",
    *BYTE_COLUMNS,
]

ENGINEERED_FEATURES = [
    "payload_zero_count",
    "payload_nonzero_count",
    "payload_byte_mean",
    "payload_byte_std",
    "payload_byte_entropy",
    "payload_unique_byte_count",
    "payload_bit_ones_count",
    "payload_bit_density",
]

COMPACT_ENGINEERED = [
    "payload_zero_count",
    "payload_byte_entropy",
    "payload_byte_std",
    "payload_bit_density",
    "payload_byte_mean",
]

PAYLOAD_FEATURES = [
    "payload_length",
    *BYTE_COLUMNS,
]

BINARY_ID_FEATURES = [
    f"id_bit_{i}" for i in range(10, -1, -1)
]

CONFIGURATIONS = {
    "E1_original_10": ORIGINAL_FEATURES,

    "E4_original_plus_compact_15": (
        ORIGINAL_FEATURES + COMPACT_ENGINEERED
    ),

    "E5_original_plus_all_engineered_18": (
        ORIGINAL_FEATURES + ENGINEERED_FEATURES
    ),

    "B1_binary_id_plus_original_payload_20": (
        BINARY_ID_FEATURES + PAYLOAD_FEATURES
    ),
}

RF_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "max_features": 3,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# SAFE METRIC FORMATTING
# ============================================================

def fmt_metric(value, digits=4):
    """
    Safely format a metric.

    Undefined metrics are displayed as N/A rather than
    causing a formatting exception.
    """
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.{digits}f}"


# ============================================================
# DATA LOADING
# ============================================================

def load_files(file_names, split_name):

    frames = []

    print(f"\nLoading {split_name} data...")

    for filename in file_names:

        path = INPUT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        required = (
            ORIGINAL_FEATURES
            + ENGINEERED_FEATURES
            + [TARGET]
        )

        missing = [
            c for c in required if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{filename}: missing columns {missing}"
            )

        if df[required].isna().any().any():
            raise ValueError(
                f"{filename}: missing values found"
            )

        if not set(df[TARGET].unique()).issubset({0, 1}):
            raise ValueError(
                f"{filename}: invalid target labels"
            )

        df["source_file"] = filename

        frames.append(df)

        print(
            f"  {filename}: {len(df):,} rows, "
            f"{int(df[TARGET].sum()):,} attacks"
        )

    result = pd.concat(frames, ignore_index=True)

    return result


# ============================================================
# BINARY CAN ID FEATURES
# ============================================================

def add_binary_id_features(df):

    result = df.copy()

    ids = result["arbitration_id"].to_numpy(
        dtype=np.int64
    )

    if (ids < 0).any() or (ids > 2047).any():
        raise ValueError("ID outside 11-bit range")

    for bit in range(10, -1, -1):

        result[f"id_bit_{bit}"] = (
            (ids >> bit) & 1
        ).astype(np.int8)

    return result


# ============================================================
# FAMILY IDENTIFICATION
# ============================================================

def identify_family(filename):

    name = Path(filename).stem.lower()

    if name.startswith("accessory"):
        return "accessory"

    if name.startswith("dos"):
        return "dos"

    if name.startswith("force-neutral"):
        return "force_neutral"

    if name.startswith("rpm"):
        return "rpm"

    if name.startswith("standstill"):
        return "standstill"

    return "unknown"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold=0.5):

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) else 0.0
    )

    result = {
        "rows": int(len(y_true)),
        "actual_attacks": int(np.sum(y_true)),
        "actual_normal": int(len(y_true) - np.sum(y_true)),
        "threshold": float(threshold),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "fpr": float(fpr),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    # PR-AUC and ROC-AUC are undefined when only one
    # class is present in the evaluated recording.

    if len(np.unique(y_true)) == 2:

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

    else:

        result["pr_auc"] = None
        result["roc_auc"] = None

    return result


# ============================================================
# PROBABILITY COMPARISON
# ============================================================

def compare_probability_scores(prediction_data):

    rows = []

    comparisons = [
        (
            "E4_original_plus_compact_15",
            "E5_original_plus_all_engineered_18",
        ),

        (
            "E4_original_plus_compact_15",
            "B1_binary_id_plus_original_payload_20",
        ),

        (
            "E5_original_plus_all_engineered_18",
            "B1_binary_id_plus_original_payload_20",
        ),
    ]

    for name_a, name_b in comparisons:

        a = prediction_data[name_a]["probabilities"]
        b = prediction_data[name_b]["probabilities"]

        correlation = spearmanr(a, b).statistic

        rows.append({
            "model_a": name_a,
            "model_b": name_b,

            "spearman_score_correlation": float(
                correlation
            ),

            "mean_absolute_probability_difference": float(
                np.mean(np.abs(a - b))
            ),

            "maximum_absolute_probability_difference": float(
                np.max(np.abs(a - b))
            ),

            "identical_probability_fraction": float(
                np.mean(a == b)
            ),

            "same_prediction_fraction_at_0_5": float(
                np.mean((a >= 0.5) == (b >= 0.5))
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL VALIDATION DIAGNOSTICS")
    print("=" * 70)

    train_data = load_files(
        TRAIN_FILES,
        "training",
    )

    train_data = add_binary_id_features(train_data)

    y_train = train_data[TARGET].to_numpy(
        dtype=np.int8
    )

    print("\nTraining rows:", f"{len(train_data):,}")
    print("Training attacks:", f"{int(y_train.sum()):,}")

    pooled_rows = []
    recording_rows = []
    prediction_data = {}

    e4_importance = None
    e4_model = None

    # --------------------------------------------------------
    # Train each candidate and evaluate all validation files.
    # --------------------------------------------------------

    for config_name, features in CONFIGURATIONS.items():

        print("\n" + "=" * 70)
        print("MODEL:", config_name)
        print("=" * 70)

        model = RandomForestClassifier(**RF_PARAMETERS)

        start = time.time()

        model.fit(
            train_data[features],
            y_train,
        )

        training_seconds = time.time() - start

        print(f"Training time: {training_seconds:.2f}s")

        all_labels = []
        all_probabilities = []
        all_sources = []

        # ----------------------------------------------------
        # Process each recording independently.
        # ----------------------------------------------------

        for filename in VALIDATION_FILES:

            path = INPUT_DIR / filename

            validation = pd.read_csv(path)

            validation = add_binary_id_features(
                validation
            )

            y_true = validation[TARGET].to_numpy(
                dtype=np.int8
            )

            probabilities = model.predict_proba(
                validation[features]
            )[:, 1]

            all_labels.append(y_true)
            all_probabilities.append(probabilities)

            all_sources.extend(
                [filename] * len(validation)
            )

            metrics = calculate_metrics(
                y_true,
                probabilities,
                threshold=0.5,
            )

            recording_rows.append({
                "configuration": config_name,
                "recording": filename,
                "attack_family": identify_family(filename),
                **metrics,
            })

            # FIX: Use fmt_metric for potentially undefined
            # metrics such as PR-AUC and ROC-AUC.

            print(
                f"  {filename}: "
                f"PR-AUC={fmt_metric(metrics['pr_auc'], 5)} | "
                f"F1={fmt_metric(metrics['f1'], 5)} | "
                f"Recall={fmt_metric(metrics['recall'], 5)} | "
                f"FPR={fmt_metric(metrics['fpr'], 6)}"
            )

        # ----------------------------------------------------
        # Pooled validation metrics.
        # ----------------------------------------------------

        y_all = np.concatenate(all_labels)
        p_all = np.concatenate(all_probabilities)
        sources = np.asarray(all_sources)

        pooled_metrics = calculate_metrics(
            y_all,
            p_all,
            threshold=0.5,
        )

        pooled_rows.append({
            "configuration": config_name,
            "feature_count": len(features),
            "training_seconds": float(training_seconds),
            **pooled_metrics,
        })

        prediction_data[config_name] = {
            "labels": y_all,
            "probabilities": p_all,
            "sources": sources,
        }

        # ----------------------------------------------------
        # Save fitted model.
        # ----------------------------------------------------

        model_path = MODEL_DIR / f"{config_name}.joblib"

        joblib.dump(
            {
                "model": model,
                "features": features,
                "target": TARGET,
                "parameters": RF_PARAMETERS,
            },
            model_path,
        )

        # ----------------------------------------------------
        # Save E4 feature importance.
        # ----------------------------------------------------

        if config_name == "E4_original_plus_compact_15":

            e4_model = model

            e4_importance = pd.DataFrame({
                "feature": features,

                "impurity_importance": (
                    model.feature_importances_
                ),

            }).sort_values(
                "impurity_importance",
                ascending=False,
            )

        del model

    # --------------------------------------------------------
    # Save pooled metrics.
    # --------------------------------------------------------

    pooled_df = pd.DataFrame(pooled_rows)

    pooled_df.to_csv(
        REPORT_DIR / "01_full_validation_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save per-recording metrics.
    # --------------------------------------------------------

    recording_df = pd.DataFrame(recording_rows)

    recording_df.to_csv(
        REPORT_DIR / "02_per_recording_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Probability comparison.
    # --------------------------------------------------------

    probability_df = compare_probability_scores(
        prediction_data
    )

    probability_df.to_csv(
        REPORT_DIR / "03_probability_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # E4 feature importance.
    # --------------------------------------------------------

    if e4_importance is not None:

        e4_importance.to_csv(
            REPORT_DIR / "04_e4_feature_importance.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Save full validation probabilities.
    # --------------------------------------------------------

    prediction_arrays = {
        "sources": prediction_data[
            "E1_original_10"
        ]["sources"],

        "labels": prediction_data[
            "E1_original_10"
        ]["labels"],
    }

    for config_name, values in prediction_data.items():

        prediction_arrays[
            config_name + "_probabilities"
        ] = values["probabilities"]

    np.savez_compressed(
        REPORT_DIR / "05_full_validation_predictions.npz",
        **prediction_arrays,
    )

    # --------------------------------------------------------
    # Metadata.
    # --------------------------------------------------------

    metadata = {
        "experiment": "Full validation diagnostics",
        "random_state": RANDOM_STATE,

        "training_rows": int(len(train_data)),
        "training_attack_rows": int(y_train.sum()),

        "validation_rows": int(
            len(prediction_data["E1_original_10"]["labels"])
        ),

        "validation_attack_rows": int(
            prediction_data["E1_original_10"]["labels"].sum()
        ),

        "training_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,

        "configurations": CONFIGURATIONS,

        "random_forest_parameters": RF_PARAMETERS,

        "decision_threshold": 0.5,
        "threshold_optimisation_performed": False,
        "official_test_partitions_used": False,
    }

    with open(
        REPORT_DIR / "experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(metadata, file, indent=4)

    # --------------------------------------------------------
    # Final output.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FULL VALIDATION DIAGNOSTICS COMPLETE")
    print("=" * 70)

    print("\nPOOLED VALIDATION METRICS")

    print(
        pooled_df[
            [
                "configuration",
                "pr_auc",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "fpr",
            ]
        ].to_string(index=False)
    )

    print("\nPROBABILITY COMPARISON")
    print(probability_df.to_string(index=False))

    print("\nE4 FEATURE IMPORTANCE")

    if e4_importance is not None:
        print(e4_importance.to_string(index=False))
    else:
        print("E4 feature importance unavailable.")

    print(f"\nReports: {REPORT_DIR}")
    print(f"Models: {MODEL_DIR}")


if __name__ == "__main__":
    main()
