
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "prepared" / "set_01" / "train_01"
)

OUTPUT_DIR = (
    ROOT / "reports" / "set_01" / "stage_6"
    / "can_id_representation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
PREVIEW_ROWS = 200

TARGET = "attack"

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

REQUIRED_COLUMNS = [
    "arbitration_id",
    *PAYLOAD_FEATURES,
    TARGET,
]


# ============================================================
# DATA LOADING
# ============================================================

def load_partition(file_names, partition_name):
    frames = []

    for file_name in file_names:
        path = DATA_DIR / file_name

        if not path.exists():
            raise FileNotFoundError(
                f"{partition_name} file not found: {path}"
            )

        print(f"Loading {partition_name}: {file_name}")

        frame = pd.read_csv(
            path,
            usecols=REQUIRED_COLUMNS,
        )

        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)

    return result


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(train_df, val_df):
    for name, frame in [
        ("training", train_df),
        ("validation", val_df),
    ]:
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name}: missing columns: {sorted(missing)}"
            )

        if frame[REQUIRED_COLUMNS].isnull().any().any():
            raise ValueError(
                f"{name}: missing values detected"
            )

        ids = frame["arbitration_id"]

        if not pd.api.types.is_numeric_dtype(ids):
            raise TypeError(
                f"{name}: arbitration_id must be numeric"
            )

        if not np.isfinite(ids).all():
            raise ValueError(
                f"{name}: non-finite arbitration IDs"
            )

        if not ((ids >= 0) & (ids <= 2047)).all():
            raise ValueError(
                f"{name}: ID outside the 11-bit numeric range"
            )

        if not (ids == np.floor(ids)).all():
            raise ValueError(
                f"{name}: non-integer arbitration IDs detected"
            )

        if not frame[TARGET].isin([0, 1]).all():
            raise ValueError(
                f"{name}: target must contain only 0 and 1"
            )

    print("\nInput validation passed.")


# ============================================================
# REPRESENTATION 1: INTEGER
# ============================================================

def integer_representation(frame):
    return frame[["arbitration_id"]].copy()


# ============================================================
# REPRESENTATION 2: 11-BIT BINARY
# ============================================================

def binary_representation(frame):
    ids = frame["arbitration_id"].to_numpy(dtype=np.uint16)

    bit_data = {}

    # Most significant bit first: bit 10 -> bit 0
    for bit_position in range(10, -1, -1):
        column_name = f"id_bit_{bit_position}"

        bit_data[column_name] = (
            (ids >> bit_position) & 1
        ).astype(np.uint8)

    return pd.DataFrame(
        bit_data,
        index=frame.index,
    )


def reconstruct_binary_ids(binary_df):
    reconstructed = np.zeros(
        len(binary_df),
        dtype=np.uint16,
    )

    for bit_position in range(10, -1, -1):
        column_name = f"id_bit_{bit_position}"

        reconstructed |= (
            binary_df[column_name].to_numpy(dtype=np.uint16)
            << bit_position
        )

    return reconstructed


# ============================================================
# REPRESENTATION 3: CATEGORICAL / ONE-HOT
# ============================================================

def fit_category_schema(train_df):
    """
    Learn the category vocabulary using training recordings only.
    """

    categories = sorted(
        train_df["arbitration_id"].unique().astype(int).tolist()
    )

    return categories


def categorical_representation(frame, categories):
    """
    One-hot encode using the training category vocabulary.

    IDs not observed during training produce an all-zero
    categorical vector.
    """

    category_columns = [
        f"id_cat_{category:03X}"
        for category in categories
    ]

    categorical_ids = pd.Categorical(
        frame["arbitration_id"].astype(int),
        categories=categories,
    )

    encoded = pd.get_dummies(
        categorical_ids,
        dtype=np.uint8,
    )

    encoded.columns = category_columns

    encoded.index = frame.index

    return encoded


# ============================================================
# REPRESENTATION VALIDATION
# ============================================================

def validate_representations(
    train_df,
    val_df,
    train_binary,
    val_binary,
    train_categorical,
    val_categorical,
    categories,
):
    print("\nValidating representations...")

    # 1. Integer representation
    train_integer = integer_representation(train_df)
    val_integer = integer_representation(val_df)

    assert np.array_equal(
        train_integer["arbitration_id"].to_numpy(),
        train_df["arbitration_id"].to_numpy(),
    )

    assert np.array_equal(
        val_integer["arbitration_id"].to_numpy(),
        val_df["arbitration_id"].to_numpy(),
    )

    print("PASS: Integer representation preserves IDs.")

    # 2. Binary representation
    train_reconstructed = reconstruct_binary_ids(train_binary)
    val_reconstructed = reconstruct_binary_ids(val_binary)

    assert np.array_equal(
        train_reconstructed,
        train_df["arbitration_id"].to_numpy(dtype=np.uint16),
    )

    assert np.array_equal(
        val_reconstructed,
        val_df["arbitration_id"].to_numpy(dtype=np.uint16),
    )

    assert set(np.unique(train_binary.to_numpy())).issubset({0, 1})
    assert set(np.unique(val_binary.to_numpy())).issubset({0, 1})

    print("PASS: Binary representation reconstructs every ID.")
    print("PASS: Binary representation contains only 0 and 1.")

    # 3. Categorical representation
    train_row_sums = train_categorical.sum(axis=1)
    val_row_sums = val_categorical.sum(axis=1)

    assert (train_row_sums == 1).all()

    known_validation_mask = val_df["arbitration_id"].isin(categories)

    assert (
        val_row_sums[known_validation_mask] == 1
    ).all()

    assert (
        val_row_sums[~known_validation_mask] == 0
    ).all()

    print("PASS: Training IDs map to exactly one category.")
    print("PASS: Known validation IDs map to exactly one category.")
    print("PASS: Unseen validation IDs map to all-zero vectors.")

    return {
        "training_unique_ids": int(
            train_df["arbitration_id"].nunique()
        ),
        "validation_unique_ids": int(
            val_df["arbitration_id"].nunique()
        ),
        "training_category_count": len(categories),
        "unseen_validation_rows": int(
            (~known_validation_mask).sum()
        ),
        "unseen_validation_unique_ids": int(
            val_df.loc[
                ~known_validation_mask,
                "arbitration_id",
            ].nunique()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("CAN ID REPRESENTATION EXPERIMENT")
    print("================================")

    train_df = load_partition(
        TRAIN_FILES,
        "training",
    )

    val_df = load_partition(
        VALIDATION_FILES,
        "validation",
    )

    validate_input(train_df, val_df)

    print("\nDataset dimensions")
    print("------------------------------")
    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")

    # Integer representation
    train_integer = integer_representation(train_df)
    val_integer = integer_representation(val_df)

    # Binary representation
    train_binary = binary_representation(train_df)
    val_binary = binary_representation(val_df)

    # Categorical vocabulary learned ONLY from training data
    categories = fit_category_schema(train_df)

    train_categorical = categorical_representation(
        train_df,
        categories,
    )

    val_categorical = categorical_representation(
        val_df,
        categories,
    )

    # Validate all representations
    validation_summary = validate_representations(
        train_df,
        val_df,
        train_binary,
        val_binary,
        train_categorical,
        val_categorical,
        categories,
    )

    # Feature dimensions
    feature_dimensions = {
        "integer": 1,
        "binary": train_binary.shape[1],
        "categorical": train_categorical.shape[1],
        "payload": len(PAYLOAD_FEATURES),
    }

    unseen_ids = sorted(
    int(x)
    for x in (
        set(val_df["arbitration_id"].astype(int).unique())
        - set(categories)
    )
)

    # Save a small preview, not the full feature matrices
    train_preview = pd.concat(
        [
            train_integer.head(PREVIEW_ROWS).reset_index(drop=True),
            train_binary.head(PREVIEW_ROWS).reset_index(drop=True),
            train_categorical.head(PREVIEW_ROWS).reset_index(drop=True),
            train_df[PAYLOAD_FEATURES + [TARGET]]
            .head(PREVIEW_ROWS)
            .reset_index(drop=True),
        ],
        axis=1,
    )

    val_preview = pd.concat(
        [
            val_integer.head(PREVIEW_ROWS).reset_index(drop=True),
            val_binary.head(PREVIEW_ROWS).reset_index(drop=True),
            val_categorical.head(PREVIEW_ROWS).reset_index(drop=True),
            val_df[PAYLOAD_FEATURES + [TARGET]]
            .head(PREVIEW_ROWS)
            .reset_index(drop=True),
        ],
        axis=1,
    )

    train_preview.to_csv(
        OUTPUT_DIR / "training_representation_preview.csv",
        index=False,
    )

    val_preview.to_csv(
        OUTPUT_DIR / "validation_representation_preview.csv",
        index=False,
    )

    metadata = {
        "experiment": "CAN arbitration ID representation validation",
        "random_state": RANDOM_STATE,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "training_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "representations": {
            "integer": {
                "description": "Original numeric arbitration ID",
                "feature_count": 1,
            },
            "binary": {
                "description": "11-bit binary representation, MSB first",
                "feature_count": 11,
                "columns": [
                    f"id_bit_{i}" for i in range(10, -1, -1)
                ],
            },
            "categorical": {
                "description": "One-hot encoding using training IDs only",
                "feature_count": len(categories),
                "categories": categories,
                "unseen_validation_ids": unseen_ids,
            },
        },
        "payload_features_retained": PAYLOAD_FEATURES,
        "target": TARGET,
        "validation": validation_summary,
        "feature_dimensions": feature_dimensions,
        "note": (
            "No model training was performed. "
            "Categorical vocabulary was fitted only on training data."
        ),
    }

    with open(
        OUTPUT_DIR / "representation_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nRepresentation dimensions")
    print("------------------------------")
    for name, count in feature_dimensions.items():
        print(f"{name}: {count} feature(s)")

    print("\nValidation summary")
    print("------------------------------")
    for key, value in validation_summary.items():
        print(f"{key}: {value}")

    print("\nUnseen validation IDs:")
    print(unseen_ids)

    print("\nREPRESENTATION VALIDATION COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
