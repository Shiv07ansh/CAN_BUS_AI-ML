
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Project paths and configuration
# ---------------------------------------------------------------------

SHARED_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = SHARED_DIR / "data" / "processed" / "set_01"
REPRESENTATIONS_DIR = PROCESSED_DIR / "representations"
SPLIT_DIR = PROCESSED_DIR / "split"

OUTPUT_DIR = SHARED_DIR / "outputs" / "datasets"

TARGET_COLUMN = "attack"
CHUNK_SIZE = 200_000

SUPPORTED_REPRESENTATIONS = {"E1", "E4", "E5", "B1"}


# ---------------------------------------------------------------------
# Manifest and input helpers
# ---------------------------------------------------------------------

def read_manifest_filenames(manifest_path):
    """Read recording filenames from a frozen split manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)
    candidates = []

    for column in manifest.columns:
        values = (
            manifest[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        filenames = values[
            values.str.lower().str.endswith(".csv")
        ].tolist()

        if filenames:
            candidates.append(filenames)

    if not candidates:
        raise ValueError(
            f"No CSV filenames found in {manifest_path}"
        )

    candidates.sort(key=len, reverse=True)
    filenames = candidates[0]

    if len(filenames) != len(set(filenames)):
        raise ValueError(
            f"Duplicate recording filenames in {manifest_path}"
        )

    return filenames


def get_split_files():
    """Return the frozen training and validation recording lists."""
    training_files = read_manifest_filenames(
        SPLIT_DIR / "training_files.csv"
    )

    validation_files = read_manifest_filenames(
        SPLIT_DIR / "validation_files.csv"
    )

    overlap = set(training_files) & set(validation_files)

    if overlap:
        raise ValueError(
            f"Training/validation overlap detected: {sorted(overlap)}"
        )

    if len(training_files) != 7:
        raise ValueError(
            f"Expected 7 training recordings, found {len(training_files)}"
        )

    if len(validation_files) != 5:
        raise ValueError(
            f"Expected 5 validation recordings, found {len(validation_files)}"
        )

    return training_files, validation_files


def get_input_path(representation, partition, filename):
    """Resolve a representation CSV path."""
    return (
        REPRESENTATIONS_DIR
        / representation
        / partition
        / filename
    )


def get_feature_columns(csv_path):
    """Return all feature columns, excluding the target."""
    columns = list(pd.read_csv(csv_path, nrows=0).columns)

    if TARGET_COLUMN not in columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' missing from {csv_path}"
        )

    feature_columns = [
        column for column in columns
        if column != TARGET_COLUMN
    ]

    if not feature_columns:
        raise ValueError(f"No feature columns found in {csv_path}")

    return feature_columns



def validate_input_schema(csv_path, expected_columns):
    """Validate feature columns and target column in the correct order."""
    actual_columns = list(pd.read_csv(csv_path, nrows=0).columns)

    expected_full_columns = expected_columns + [TARGET_COLUMN]

    if actual_columns != expected_full_columns:
        raise ValueError(
            f"Schema mismatch in {csv_path}\n"
            f"Expected: {expected_full_columns}\n"
            f"Found:    {actual_columns}"
        )



def validate_chunk(frame, feature_columns, filename):
    """Check feature values and binary target labels."""
    features = frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if features.isna().any().any():
        raise ValueError(
            f"Missing or non-numeric feature values in {filename}"
        )

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(
            f"Infinite feature values in {filename}"
        )

    labels = pd.to_numeric(
        frame[TARGET_COLUMN],
        errors="coerce"
    )

    if labels.isna().any():
        raise ValueError(
            f"Missing or non-numeric target labels in {filename}"
        )

    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(
            f"Target contains values outside 0/1 in {filename}"
        )

    return features, labels


# ---------------------------------------------------------------------
# Training-only scaler fitting
# ---------------------------------------------------------------------

def fit_training_scaler(
    representation,
    training_files,
    feature_columns
):
    """
    Fit StandardScaler incrementally using training recordings only.

    Validation recordings are never used to estimate scaling statistics.
    """
    scaler = StandardScaler()

    rows_seen = 0

    for filename in training_files:
        csv_path = get_input_path(
            representation,
            "training",
            filename
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Training representation missing: {csv_path}"
            )

        validate_input_schema(csv_path, feature_columns)

        for frame in pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE
        ):
            features, _ = validate_chunk(
                frame,
                feature_columns,
                str(csv_path)
            )

            scaler.partial_fit(features)
            rows_seen += len(features)

        print(f"Scaler fit: {filename}")

    if rows_seen == 0:
        raise ValueError("No training rows were available for scaler fitting")

    print(f"Scaler fitted on {rows_seen:,} training rows")

    return scaler


# ---------------------------------------------------------------------
# Transform and save recordings
# ---------------------------------------------------------------------

def transform_partition(
    representation,
    partition,
    filenames,
    feature_columns,
    scaler,
    output_directory
):
    """Transform each recording and preserve its target column."""
    output_directory.mkdir(parents=True, exist_ok=True)

    total_rows = 0

    for filename in filenames:
        input_path = get_input_path(
            representation,
            partition,
            filename
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input representation missing: {input_path}"
            )

        validate_input_schema(input_path, feature_columns)

        output_path = output_directory / filename

        if output_path.exists():
            output_path.unlink()

        rows_written = 0

        for frame in pd.read_csv(
            input_path,
            chunksize=CHUNK_SIZE
        ):
            features, labels = validate_chunk(
                frame,
                feature_columns,
                str(input_path)
            )

            if scaler is not None:
                transformed = scaler.transform(features)
                output_features = pd.DataFrame(
                    transformed,
                    columns=feature_columns
                )
            else:
                output_features = features.reset_index(drop=True)

            output_frame = output_features.copy()
            output_frame[TARGET_COLUMN] = labels.to_numpy()

            output_frame.to_csv(
                output_path,
                mode="a",
                header=(rows_written == 0),
                index=False
            )

            rows_written += len(output_frame)

        total_rows += rows_written

        print(
            f"{partition}: {filename} "
            f"({rows_written:,} rows)"
        )

    return total_rows


# ---------------------------------------------------------------------
# Public preprocessing function
# ---------------------------------------------------------------------

def preprocess_representation(
    representation,
    model_name,
    scale_features
):
    """
    Preprocess one representation for one model.

    Parameters
    ----------
    representation : str
        One of E1, E4, E5, B1.

    model_name : str
        Output model identifier, e.g. logistic_regression.

    scale_features : bool
        True for models requiring StandardScaler.
        False to preserve original feature values.
    """
    representation = representation.upper()

    if representation not in SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"Unsupported representation: {representation}. "
            f"Choose from {sorted(SUPPORTED_REPRESENTATIONS)}"
        )

    if not model_name or not model_name.strip():
        raise ValueError("model_name cannot be empty")

    training_files, validation_files = get_split_files()

    first_training_file = get_input_path(
        representation,
        "training",
        training_files[0]
    )

    if not first_training_file.exists():
        raise FileNotFoundError(
            f"Training representation missing: {first_training_file}"
        )

    feature_columns = get_feature_columns(first_training_file)

    output_base = (
        OUTPUT_DIR
        / model_name
        / representation
    )

    scaler = None

    if scale_features:
        print(
            f"\nFitting StandardScaler for "
            f"{model_name} / {representation}"
        )

        scaler = fit_training_scaler(
            representation,
            training_files,
            feature_columns
        )

        scaler_path = output_base / "scaler.joblib"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(scaler, scaler_path)

        print(f"Scaler saved: {scaler_path}")

    else:
        print(
            f"\nScaling disabled for "
            f"{model_name} / {representation}"
        )

    training_rows = transform_partition(
        representation=representation,
        partition="training",
        filenames=training_files,
        feature_columns=feature_columns,
        scaler=scaler,
        output_directory=output_base / "training"
    )

    validation_rows = transform_partition(
        representation=representation,
        partition="validation",
        filenames=validation_files,
        feature_columns=feature_columns,
        scaler=scaler,
        output_directory=output_base / "validation"
    )

    metadata = {
        "model": model_name,
        "representation": representation,
        "scaling_enabled": scale_features,
        "scaler": "StandardScaler" if scale_features else None,
        "scaler_fit_data": "training recordings only" if scale_features else None,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "training_recordings": training_files,
        "validation_recordings": validation_files,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "chunk_size": CHUNK_SIZE,
    }

    metadata_path = output_base / "preprocessing_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nPreprocessing complete")
    print(f"Model: {model_name}")
    print(f"Representation: {representation}")
    print(f"Training rows: {training_rows:,}")
    print(f"Validation rows: {validation_rows:,}")
    print(f"Output: {output_base}")

    return metadata
