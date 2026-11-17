
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
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

INPUT_DIR = (
    ROOT / "data" / "engineered" / "set_01" / "train_01"
)

REPORT_DIR = (
    ROOT / "reports" / "set_01" / "stage_6"
    / "payload_experiment_pipeline"
)

MODEL_REPORT_DIR = REPORT_DIR / "model_comparison"

RANDOM_STATE = 42
VALIDATION_SAMPLE_SIZE = 50_000

# Keep these exactly aligned with the frozen Stage 4 split.
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

TARGET = "attack"

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

COMPACT_ENGINEERED_FEATURES = [
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

ALL_REQUIRED = (
    ORIGINAL_FEATURES
    + ENGINEERED_FEATURES
    + [TARGET]
)

# Same absolute max_features value across all configurations,
# matching the previous controlled CAN ID comparison.
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
# FEATURE CONFIGURATIONS
# ============================================================

FEATURE_CONFIGURATIONS = {
    "E1_original_10": ORIGINAL_FEATURES,

    "E2_engineered_8": ENGINEERED_FEATURES,

    "E3_compact_engineered_5": COMPACT_ENGINEERED_FEATURES,

    "E4_original_plus_compact_15": (
        ORIGINAL_FEATURES + COMPACT_ENGINEERED_FEATURES
    ),

    "E5_original_plus_all_engineered_18": (
        ORIGINAL_FEATURES + ENGINEERED_FEATURES
    ),

    "B1_binary_id_plus_original_payload_20": (
        BINARY_ID_FEATURES + PAYLOAD_FEATURES
    ),

    "B2_binary_id_plus_all_payload_features_28": (
        BINARY_ID_FEATURES
        + PAYLOAD_FEATURES
        + ENGINEERED_FEATURES
    ),
}


# ============================================================
# DATA LOADING
# ============================================================

def load_recordings(file_names, split_name):
    frames = []

    print(f"\nLoading {split_name} recordings...")

    for filename in file_names:
        path = INPUT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing engineered recording: {path}"
            )

        df = pd.read_csv(path)

        missing = [
            column for column in ALL_REQUIRED
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{filename} missing columns: {missing}"
            )

        if df[ALL_REQUIRED].isna().any().any():
            raise ValueError(
                f"Missing values detected in {filename}"
            )

        if not set(df[TARGET].unique()).issubset({0, 1}):
            raise ValueError(
                f"Unexpected labels in {filename}"
            )

        df["source_file"] = filename

        frames.append(df)

        print(
            f"  {filename}: {len(df):,} rows, "
            f"{int(df[TARGET].sum()):,} attacks"
        )

    result = pd.concat(frames, ignore_index=True)

    if not np.isfinite(
        result[ALL_REQUIRED].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            f"Non-finite values in {split_name} data"
        )

    return result


# ============================================================
# BINARY CAN ID REPRESENTATION
# ============================================================

def add_binary_id_features(data):
    result = data.copy()

    ids = result["arbitration_id"].to_numpy(
        dtype=np.int64
    )

    if (ids < 0).any() or (ids > 2047).any():
        raise ValueError(
            "CAN arbitration ID outside 11-bit range"
        )

    for bit in range(10, -1, -1):
        result[f"id_bit_{bit}"] = (
            (ids >> bit) & 1
        ).astype(np.int8)

    return result


# ============================================================
# 1. ATTACK-FAMILY AND RECORDING-LEVEL ANALYSIS
# ============================================================

def identify_attack_family(filename):
    name = Path(filename).stem.lower()

    if name.startswith("attack-free"):
        return "normal_recording"

    if name.startswith("force-neutral"):
        return "force_neutral"

    if name.startswith("standstill"):
        return "standstill"

    if name.startswith("accessory"):
        return "accessory"

    if name.startswith("dos"):
        return "dos"

    if name.startswith("rpm"):
        return "rpm"

    return "unknown"


def distribution_analysis(train_data):
    features_to_analyse = [
        "arbitration_id",
        "payload_length",
        *ENGINEERED_FEATURES,
    ]

    data = train_data.copy()

    data["attack_family"] = data["source_file"].map(
        identify_attack_family
    )

    # --------------------------------------------------------
    # Attack family distributions
    # --------------------------------------------------------

    family_rows = []

    attack_data = data[data[TARGET] == 1]

    for family, group in attack_data.groupby(
        "attack_family"
    ):
        for feature in features_to_analyse:
            values = group[feature]

            family_rows.append({
                "attack_family": family,
                "feature": feature,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "min": float(values.min()),
                "max": float(values.max()),
            })

    pd.DataFrame(family_rows).to_csv(
        REPORT_DIR / "01_attack_family_distributions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Recording-level class distributions
    # --------------------------------------------------------

    recording_rows = []

    for filename, group in data.groupby("source_file"):
        family = identify_attack_family(filename)

        for label, class_group in group.groupby(TARGET):
            class_name = (
                "attack" if label == 1 else "normal"
            )

            for feature in features_to_analyse:
                values = class_group[feature]

                recording_rows.append({
                    "recording": filename,
                    "attack_family": family,
                    "class": class_name,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                })

    pd.DataFrame(recording_rows).to_csv(
        REPORT_DIR / "02_recording_level_distributions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Per-recording attack concentration
    # --------------------------------------------------------

    concentration_rows = []

    for filename, group in data.groupby("source_file"):
        attack_count = int(group[TARGET].sum())

        concentration_rows.append({
            "recording": filename,
            "attack_family": identify_attack_family(filename),
            "total_rows": int(len(group)),
            "attack_rows": attack_count,
            "normal_rows": int(len(group) - attack_count),
            "attack_prevalence": float(group[TARGET].mean()),
        })

    pd.DataFrame(concentration_rows).to_csv(
        REPORT_DIR / "03_recording_attack_concentration.csv",
        index=False,
    )

    print("\nDistribution analysis completed.")


# ============================================================
# 2. FIXED VALIDATION SAMPLE
# ============================================================

def create_validation_sample(validation_data):
    if len(validation_data) < VALIDATION_SAMPLE_SIZE:
        raise ValueError(
            "Validation data smaller than requested sample"
        )

    sample, _ = train_test_split(
        validation_data,
        train_size=VALIDATION_SAMPLE_SIZE,
        stratify=validation_data[TARGET],
        random_state=RANDOM_STATE,
    )

    sample = sample.reset_index(drop=True)

    print("\nFixed validation sample:")
    print(f"  Rows: {len(sample):,}")
    print(f"  Attacks: {int(sample[TARGET].sum()):,}")
    print(
        f"  Attack prevalence: "
        f"{sample[TARGET].mean() * 100:.4f}%"
    )

    return sample


# ============================================================
# 3. METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0.0
    )

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


def find_best_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    denominator = precision[:-1] + recall[:-1]

    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_index = int(np.argmax(f1_values))

    return float(thresholds[best_index])


# ============================================================
# 4. MODEL COMPARISON
# ============================================================

def run_model_comparison(train_data, validation_sample):
    results = []
    threshold_results = []

    y_train = train_data[TARGET].to_numpy(dtype=np.int8)
    y_validation = validation_sample[TARGET].to_numpy(
        dtype=np.int8
    )

    for config_name, features in FEATURE_CONFIGURATIONS.items():
        print("\n" + "=" * 65)
        print(f"MODEL CONFIGURATION: {config_name}")
        print("=" * 65)

        print(f"Number of features: {len(features)}")

        missing_train = [
            feature for feature in features
            if feature not in train_data.columns
        ]

        missing_validation = [
            feature for feature in features
            if feature not in validation_sample.columns
        ]

        if missing_train or missing_validation:
            raise ValueError(
                f"Missing features for {config_name}: "
                f"train={missing_train}, "
                f"validation={missing_validation}"
            )

        x_train = train_data[features]
        x_validation = validation_sample[features]

        model = RandomForestClassifier(
            **RF_PARAMETERS
        )

        start = time.time()

        model.fit(x_train, y_train)

        training_seconds = time.time() - start

        probabilities = model.predict_proba(
            x_validation
        )[:, 1]

        pr_auc = average_precision_score(
            y_validation,
            probabilities,
        )

        roc_auc = roc_auc_score(
            y_validation,
            probabilities,
        )

        default_metrics = calculate_metrics(
            y_validation,
            probabilities,
            0.5,
        )

        # Exploratory only: threshold chosen and evaluated on
        # the same validation sample. Do not treat as unbiased
        # final performance.
        best_threshold = find_best_f1_threshold(
            y_validation,
            probabilities,
        )

        best_metrics = calculate_metrics(
            y_validation,
            probabilities,
            best_threshold,
        )

        results.append({
            "configuration": config_name,
            "feature_count": len(features),
            "features": ";".join(features),
            "training_rows": len(train_data),
            "validation_rows": len(validation_sample),
            "validation_attacks": int(y_validation.sum()),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "training_seconds": float(training_seconds),
            **{
                f"default_{key}": value
                for key, value in default_metrics.items()
            },
        })

        threshold_results.append({
            "configuration": config_name,
            "best_f1_threshold": best_threshold,
            "best_f1": best_metrics["f1"],
            "best_precision": best_metrics["precision"],
            "best_recall": best_metrics["recall"],
            "best_fpr": best_metrics["fpr"],
            "best_tp": best_metrics["tp"],
            "best_fp": best_metrics["fp"],
            "best_tn": best_metrics["tn"],
            "best_fn": best_metrics["fn"],
            "threshold_selection_note": (
                "Exploratory: selected and evaluated on "
                "the same validation sample"
            ),
        })

        print(f"Training time: {training_seconds:.2f}s")
        print(f"PR-AUC: {pr_auc:.6f}")
        print(f"ROC-AUC: {roc_auc:.6f}")
        print(
            f"Default threshold F1: "
            f"{default_metrics['f1']:.6f}"
        )
        print(
            f"Default precision: "
            f"{default_metrics['precision']:.6f}"
        )
        print(
            f"Default recall: "
            f"{default_metrics['recall']:.6f}"
        )
        print(
            f"Default FPR: "
            f"{default_metrics['fpr']:.6f}"
        )
        print(
            f"Exploratory best-F1 threshold: "
            f"{best_threshold:.6f}"
        )
        print(
            f"Exploratory best F1: "
            f"{best_metrics['f1']:.6f}"
        )

        del model

    results_df = pd.DataFrame(results).sort_values(
        "pr_auc",
        ascending=False,
    )

    thresholds_df = pd.DataFrame(threshold_results).sort_values(
        "best_f1",
        ascending=False,
    )

    results_df.to_csv(
        MODEL_REPORT_DIR / "01_model_comparison.csv",
        index=False,
    )

    thresholds_df.to_csv(
        MODEL_REPORT_DIR / "02_threshold_comparison.csv",
        index=False,
    )

    return results_df, thresholds_df


# ============================================================
# 5. EXPERIMENT METADATA
# ============================================================

def save_metadata(train_data, validation_data, sample):
    metadata = {
        "experiment": "Payload feature and CAN ID comparison",
        "random_state": RANDOM_STATE,
        "validation_sample_size": len(sample),
        "validation_sample_method": (
            "Stratified train_test_split, random_state=42"
        ),
        "training_rows": len(train_data),
        "training_attack_rows": int(
            train_data[TARGET].sum()
        ),
        "validation_recording_rows": len(validation_data),
        "validation_recording_attack_rows": int(
            validation_data[TARGET].sum()
        ),
        "validation_sample_attack_rows": int(
            sample[TARGET].sum()
        ),
        "training_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "feature_configurations": {
            name: features
            for name, features in FEATURE_CONFIGURATIONS.items()
        },
        "random_forest_parameters": RF_PARAMETERS,
        "validation_threshold_optimisation": (
            "Exploratory only; threshold selected and "
            "evaluated on the same validation sample"
        ),
        "official_test_partitions_used": False,
        "feature_selection_performed": False,
    }

    with open(
        REPORT_DIR / "experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)


# ============================================================
# MAIN
# ============================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PAYLOAD FEATURE AND CAN ID EXPERIMENT PIPELINE")
    print("=" * 70)

    train_data = load_recordings(
        TRAIN_FILES,
        "training",
    )

    validation_data = load_recordings(
        VALIDATION_FILES,
        "validation",
    )

    # Add the 11-bit representation to both splits.
    train_data = add_binary_id_features(train_data)
    validation_data = add_binary_id_features(validation_data)

    print("\n[1/4] Attack-family and recording analysis...")
    distribution_analysis(train_data)

    print("\n[2/4] Creating fixed validation sample...")
    validation_sample = create_validation_sample(
        validation_data
    )

    print("\n[3/4] Running model configurations...")
    results, thresholds = run_model_comparison(
        train_data,
        validation_sample,
    )

    print("\n[4/4] Saving experiment metadata...")
    save_metadata(
        train_data,
        validation_data,
        validation_sample,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT PIPELINE COMPLETE")
    print("=" * 70)

    print("\nMODEL RANKING BY PR-AUC")
    print(
        results[
            [
                "configuration",
                "feature_count",
                "pr_auc",
                "default_f1",
                "default_precision",
                "default_recall",
                "default_fpr",
            ]
        ].to_string(index=False)
    )

    print("\nMODEL RANKING BY EXPLORATORY BEST F1")
    print(
        thresholds[
            [
                "configuration",
                "best_f1_threshold",
                "best_f1",
                "best_precision",
                "best_recall",
                "best_fpr",
            ]
        ].to_string(index=False)
    )

    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
