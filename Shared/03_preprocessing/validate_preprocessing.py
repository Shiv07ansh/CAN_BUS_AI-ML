
from pathlib import Path
from itertools import zip_longest
import json

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SHARED_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = SHARED_DIR / "data" / "processed" / "set_01"

REPRESENTATIONS_DIR = PROCESSED_DIR / "representations"
SPLIT_DIR = PROCESSED_DIR / "split"

OUTPUT_DIR = SHARED_DIR / "outputs" / "datasets"
REPORT_DIR = PROCESSED_DIR / "validation"

CHUNK_SIZE = 200_000
TARGET_COLUMN = "attack"

REPRESENTATIONS = ["E1", "E4", "E5", "B1"]

MODEL_CONFIGS = {
    "logistic_regression": True,
    "linear_svm": True,
    "random_forest": False,
    "decision_tree": False,
    "naive_bayes": False,
}

PARTITIONS = ["training", "validation"]

# Numerical tolerance for independently recomputed StandardScaler output.
RTOL = 1e-6
ATOL = 1e-7


# ---------------------------------------------------------------------
# Report collection
# ---------------------------------------------------------------------

record_results = []
errors = []


def add_error(scope, message):
    errors.append({
        "scope": scope,
        "message": message,
    })


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------

def read_manifest(manifest_path):
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

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
            f"Duplicate filenames in {manifest_path}"
        )

    return filenames


def get_manifests():
    training_files = read_manifest(
        SPLIT_DIR / "training_files.csv"
    )

    validation_files = read_manifest(
        SPLIT_DIR / "validation_files.csv"
    )

    if len(training_files) != 7:
        raise ValueError(
            f"Expected 7 training files, found {len(training_files)}"
        )

    if len(validation_files) != 5:
        raise ValueError(
            f"Expected 5 validation files, found {len(validation_files)}"
        )

    overlap = set(training_files) & set(validation_files)

    if overlap:
        raise ValueError(
            f"Training/validation overlap: {sorted(overlap)}"
        )

    return {
        "training": training_files,
        "validation": validation_files,
    }


# ---------------------------------------------------------------------
# File and schema helpers
# ---------------------------------------------------------------------

def source_path(representation, partition, filename):
    return (
        REPRESENTATIONS_DIR
        / representation
        / partition
        / filename
    )


def output_path(model, representation, partition, filename):
    return (
        OUTPUT_DIR
        / model
        / representation
        / partition
        / filename
    )


def read_columns(path):
    return list(pd.read_csv(path, nrows=0).columns)


def validate_schema(source_columns, output_columns, scope):
    if source_columns != output_columns:
        add_error(
            scope,
            f"Column mismatch. Source={source_columns}; "
            f"Output={output_columns}"
        )
        return False

    if TARGET_COLUMN not in source_columns:
        add_error(scope, "Source is missing attack target column")
        return False

    return True


# ---------------------------------------------------------------------
# Chunk validation
# ---------------------------------------------------------------------

def validate_numeric(frame, feature_columns, scope):
    values = frame[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if values.isna().any().any():
        add_error(scope, "Missing or non-numeric feature values")
        return None

    array = values.to_numpy()

    if not np.isfinite(array).all():
        add_error(scope, "Infinite feature values")
        return None

    return array


def validate_labels(source_frame, output_frame, scope):
    source_labels = pd.to_numeric(
        source_frame[TARGET_COLUMN],
        errors="coerce"
    )

    output_labels = pd.to_numeric(
        output_frame[TARGET_COLUMN],
        errors="coerce"
    )

    if source_labels.isna().any() or output_labels.isna().any():
        add_error(scope, "Missing or invalid target labels")
        return False

    if not set(source_labels.unique()).issubset({0, 1}):
        add_error(scope, "Source contains target values outside 0/1")
        return False

    if not np.array_equal(
        source_labels.to_numpy(),
        output_labels.to_numpy()
    ):
        add_error(scope, "Target labels or row order differ from source")
        return False

    return True


# ---------------------------------------------------------------------
# Scaler and metadata validation
# ---------------------------------------------------------------------

def load_configuration(model, representation, should_scale):
    base = OUTPUT_DIR / model / representation

    metadata_path = base / "preprocessing_metadata.json"
    scaler_path = base / "scaler.joblib"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata: {metadata_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("model") != model:
        raise ValueError("Metadata model name mismatch")

    if metadata.get("representation") != representation:
        raise ValueError("Metadata representation mismatch")

    if metadata.get("scaling_enabled") is not should_scale:
        raise ValueError("Metadata scaling flag mismatch")

    scaler = None

    if should_scale:
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Missing scaler: {scaler_path}"
            )

        scaler = joblib.load(scaler_path)

        if metadata.get("scaler") != "StandardScaler":
            raise ValueError("Incorrect scaler recorded in metadata")

        if metadata.get("scaler_fit_data") != "training recordings only":
            raise ValueError("Scaler fitting scope is not recorded correctly")

    else:
        if scaler_path.exists():
            raise ValueError(
                "Unexpected scaler found for an unscaled model"
            )

        if metadata.get("scaler") is not None:
            raise ValueError(
                "Metadata unexpectedly records a scaler"
            )

    return metadata, scaler


# ---------------------------------------------------------------------
# Validate one recording
# ---------------------------------------------------------------------

def validate_recording(
    model,
    representation,
    partition,
    filename,
    feature_columns,
    scaler,
    should_scale
):
    scope = f"{model}/{representation}/{partition}/{filename}"

    src = source_path(representation, partition, filename)
    dst = output_path(model, representation, partition, filename)

    if not src.exists():
        add_error(scope, f"Missing source file: {src}")
        return None

    if not dst.exists():
        add_error(scope, f"Missing preprocessed file: {dst}")
        return None

    source_columns = read_columns(src)
    output_columns = read_columns(dst)

    if not validate_schema(source_columns, output_columns, scope):
        return None

    actual_features = [
        column for column in source_columns
        if column != TARGET_COLUMN
    ]

    if actual_features != feature_columns:
        add_error(scope, "Feature columns differ from metadata")
        return None

    source_iterator = pd.read_csv(
        src,
        chunksize=CHUNK_SIZE
    )

    output_iterator = pd.read_csv(
        dst,
        chunksize=CHUNK_SIZE
    )

    sentinel = object()
    total_rows = 0
    chunk_number = 0

    for source_frame, output_frame in zip_longest(
        source_iterator,
        output_iterator,
        fillvalue=sentinel
    ):
        chunk_number += 1
        chunk_scope = f"{scope}/chunk_{chunk_number}"

        if source_frame is sentinel or output_frame is sentinel:
            add_error(
                scope,
                "Source and output contain different numbers of chunks"
            )
            break

        if len(source_frame) != len(output_frame):
            add_error(
                chunk_scope,
                "Source and output chunk row counts differ"
            )
            break

        source_values = validate_numeric(
            source_frame,
            feature_columns,
            chunk_scope + "/source"
        )

        output_values = validate_numeric(
            output_frame,
            feature_columns,
            chunk_scope + "/output"
        )

        if source_values is None or output_values is None:
            break

        labels_ok = validate_labels(
            source_frame,
            output_frame,
            chunk_scope
        )

        if not labels_ok:
            break

        if should_scale:
            expected_values = scaler.transform(source_values)

            if not np.allclose(
                output_values,
                expected_values,
                rtol=RTOL,
                atol=ATOL
            ):
                add_error(
                    chunk_scope,
                    "Saved scaled features do not match "
                    "the saved StandardScaler transformation"
                )
                break

        else:
            if not np.array_equal(source_values, output_values):
                add_error(
                    chunk_scope,
                    "Unscaled features differ from source values"
                )
                break

        total_rows += len(source_frame)

    if total_rows == 0:
        add_error(scope, "No rows were validated")
        return None

    return total_rows


# ---------------------------------------------------------------------
# Validate one model-representation configuration
# ---------------------------------------------------------------------

def validate_configuration(
    model,
    representation,
    should_scale,
    manifests
):
    print(f"\nChecking {model} / {representation}")

    scope = f"{model}/{representation}"
    base = OUTPUT_DIR / model / representation

    try:
        metadata, scaler = load_configuration(
            model,
            representation,
            should_scale
        )
    except Exception as exc:
        add_error(scope, str(exc))
        return

    source_first_file = source_path(
        representation,
        "training",
        manifests["training"][0]
    )

    if not source_first_file.exists():
        add_error(scope, "First training source file is missing")
        return

    feature_columns = [
        column for column in read_columns(source_first_file)
        if column != TARGET_COLUMN
    ]

    if metadata.get("feature_columns") != feature_columns:
        add_error(scope, "Metadata feature columns mismatch")

    if scaler is not None:
        if len(scaler.mean_) != len(feature_columns):
            add_error(scope, "Scaler feature count mismatch")

    total_rows = {
        "training": 0,
        "validation": 0,
    }

    for partition in PARTITIONS:
        expected_files = manifests[partition]

        expected_directory = base / partition

        if not expected_directory.exists():
            add_error(scope, f"Missing output directory: {expected_directory}")
            continue

        actual_files = {
            path.name
            for path in expected_directory.glob("*.csv")
        }

        if actual_files != set(expected_files):
            missing = sorted(set(expected_files) - actual_files)
            unexpected = sorted(actual_files - set(expected_files))

            add_error(
                scope,
                f"{partition} file coverage mismatch. "
                f"Missing={missing}; unexpected={unexpected}"
            )

        for filename in expected_files:
            rows = validate_recording(
                model=model,
                representation=representation,
                partition=partition,
                filename=filename,
                feature_columns=feature_columns,
                scaler=scaler,
                should_scale=should_scale
            )

            if rows is not None:
                total_rows[partition] += rows

    # Check metadata row counts.
    for partition in PARTITIONS:
        metadata_key = f"{partition}_rows"

        if metadata.get(metadata_key) != total_rows[partition]:
            add_error(
                scope,
                f"Metadata {metadata_key} mismatch: "
                f"metadata={metadata.get(metadata_key)}, "
                f"verified={total_rows[partition]}"
            )

    # Check the scaler was fitted on exactly all training rows.
    if scaler is not None:
        scaler_rows = scaler.n_samples_seen_

        if isinstance(scaler_rows, np.ndarray):
            scaler_rows = scaler_rows.tolist()

        if not np.isscalar(scaler_rows):
            add_error(
                scope,
                "Unexpected per-feature scaler sample counts"
            )
        elif int(scaler_rows) != total_rows["training"]:
            add_error(
                scope,
                f"Scaler fit row count mismatch: "
                f"{scaler_rows} vs {total_rows['training']}"
            )

    config_error_count = sum(
        1 for error in errors
        if error["scope"].startswith(scope)
    )

    status = "PASS" if config_error_count == 0 else "FAIL"

    record_results.append({
        "model": model,
        "representation": representation,
        "scaling": should_scale,
        "training_rows": total_rows["training"],
        "validation_rows": total_rows["validation"],
        "status": status,
        "errors": config_error_count,
    })

    print(
        f"  Training: {total_rows['training']:,} | "
        f"Validation: {total_rows['validation']:,} | "
        f"{status}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MODEL PREPROCESSING VALIDATION")
    print("=" * 70)

    try:
        manifests = get_manifests()
    except Exception as exc:
        print(f"FATAL: Could not read split manifests: {exc}")
        return

    print(f"Training recordings: {len(manifests['training'])}")
    print(f"Validation recordings: {len(manifests['validation'])}")

    for model, should_scale in MODEL_CONFIGS.items():
        for representation in REPRESENTATIONS:
            validate_configuration(
                model,
                representation,
                should_scale,
                manifests
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(record_results)

    results_path = REPORT_DIR / "preprocessing_validation_report.csv"
    errors_path = REPORT_DIR / "preprocessing_validation_errors.csv"
    summary_path = REPORT_DIR / "preprocessing_validation_summary.json"

    results_df.to_csv(results_path, index=False)

    pd.DataFrame(
        errors,
        columns=["scope", "message"]
    ).to_csv(errors_path, index=False)

    summary = {
        "configurations_expected": (
            len(MODEL_CONFIGS) * len(REPRESENTATIONS)
        ),
        "configurations_checked": len(record_results),
        "passed_configurations": sum(
            result["status"] == "PASS"
            for result in record_results
        ),
        "failed_configurations": sum(
            result["status"] == "FAIL"
            for result in record_results
        ),
        "error_count": len(errors),
        "status": "PASSED" if not errors else "FAILED",
        "knn_included": False,
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Configurations expected: {summary['configurations_expected']}")
    print(f"Configurations checked:  {summary['configurations_checked']}")
    print(f"Passed:                  {summary['passed_configurations']}")
    print(f"Failed:                  {summary['failed_configurations']}")
    print(f"Errors:                  {summary['error_count']}")

    print(f"\nReport: {results_path}")
    print(f"Errors: {errors_path}")
    print(f"Summary: {summary_path}")

    if errors:
        print("\nPREPROCESSING VALIDATION FAILED")
    else:
        print("\nPREPROCESSING VALIDATION PASSED")


if __name__ == "__main__":
    main()
