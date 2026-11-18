
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

SHARED_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = SHARED_DIR / "data" / "processed" / "set_01"

REPRESENTATIONS_DIR = PROCESSED_DIR / "representations"
SPLIT_DIR = PROCESSED_DIR / "split"
REPORT_DIR = PROCESSED_DIR / "validation"

CHUNK_SIZE = 200_000
TARGET_COLUMN = "attack"

REPRESENTATIONS = ["E1", "E4", "E5", "B1"]

TRAINING_PARTITION = "training"
VALIDATION_PARTITION = "validation"

TEST_PARTITIONS = [
    "test_01_known_vehicle_known_attack",
    "test_02_unknown_vehicle_known_attack",
    "test_03_known_vehicle_unknown_attack",
    "test_04_unknown_vehicle_unknown_attack",
]

BASE_FEATURES = [
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

E4_ENGINEERED_FEATURES = [
    "payload_zero_count",
    "payload_byte_entropy",
    "payload_byte_std",
    "payload_bit_density",
    "payload_byte_mean",
]

EXPECTED_COLUMNS = {
    "E1": BASE_FEATURES + [TARGET_COLUMN],
    "E4": BASE_FEATURES + E4_ENGINEERED_FEATURES + [TARGET_COLUMN],
    "E5": BASE_FEATURES + ENGINEERED_FEATURES + [TARGET_COLUMN],
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

errors = []
results = []


def record_error(scope, message):
    errors.append({
        "scope": scope,
        "message": message,
    })


def read_manifest_filenames(manifest_path):
    """Find the manifest column containing recording CSV filenames."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)

    candidates = []

    for column in manifest.columns:
        values = (
            manifest[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        csv_values = values[values.str.lower().str.endswith(".csv")]

        if len(csv_values) > 0:
            candidates.append((len(csv_values), csv_values.tolist()))

    if not candidates:
        raise ValueError(
            f"No CSV filenames found in manifest: {manifest_path}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    filenames = candidates[0][1]

    if len(filenames) != len(set(filenames)):
        raise ValueError(
            f"Duplicate filenames found in manifest: {manifest_path}"
        )

    return set(filenames)


def get_csv_files(directory):
    if not directory.exists():
        raise FileNotFoundError(f"Missing directory: {directory}")

    return {
        path.name
        for path in directory.glob("*.csv")
        if path.is_file()
    }


def get_headers(path):
    return list(pd.read_csv(path, nrows=0).columns)


def validate_header(representation, columns, scope):
    if TARGET_COLUMN not in columns:
        record_error(scope, f"{representation}: missing target column")

    if len(columns) != len(set(columns)):
        record_error(scope, f"{representation}: duplicate column names")

    if representation in EXPECTED_COLUMNS:
        expected = EXPECTED_COLUMNS[representation]

        if columns != expected:
            record_error(
                scope,
                f"{representation}: unexpected column order/schema. "
                f"Expected {expected}, found {columns}"
            )

    elif representation == "B1":
        expected_non_id_features = [
            "payload_length",
            "byte_0",
            "byte_1",
            "byte_2",
            "byte_3",
            "byte_4",
            "byte_5",
            "byte_6",
            "byte_7",
            TARGET_COLUMN,
        ]

        missing = [
            column for column in expected_non_id_features
            if column not in columns
        ]

        if missing:
            record_error(
                scope,
                f"B1: missing expected columns: {missing}"
            )

        bit_columns = [
            column for column in columns
            if column not in expected_non_id_features
        ]

        if len(bit_columns) != 11:
            record_error(
                scope,
                f"B1: expected 11 CAN-ID bit columns, found "
                f"{len(bit_columns)}: {bit_columns}"
            )

        # The generator is expected to store bits MSB-first.
        # Preserve their CSV column order for reconstruction.
        if len(bit_columns) == 11:
            suffixes = []

            for column in bit_columns:
                match = re.search(r"(\d+)$", column)
                if match:
                    suffixes.append(int(match.group(1)))

            if suffixes and sorted(suffixes) != list(range(11)):
                record_error(
                    scope,
                    "B1: CAN-ID bit column suffixes do not cover 0..10"
                )

    return columns


def validate_numeric_chunk(frame, representation, scope):
    feature_columns = [
        column for column in frame.columns
        if column != TARGET_COLUMN
    ]

    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce")

        if values.isna().any():
            record_error(
                scope,
                f"{representation}: non-numeric or missing values "
                f"in {column}"
            )
            continue

        if not np.isfinite(values.to_numpy()).all():
            record_error(
                scope,
                f"{representation}: infinite values in {column}"
            )


def validate_labels(labels, representation, scope):
    values = pd.to_numeric(labels, errors="coerce")

    if values.isna().any():
        record_error(
            scope,
            f"{representation}: missing or non-numeric target labels"
        )
        return

    unique_values = set(values.unique())

    if not unique_values.issubset({0, 1}):
        record_error(
            scope,
            f"{representation}: target contains values outside 0/1: "
            f"{sorted(unique_values)}"
        )


def validate_b1_id(frame_b1, frame_e1, bit_columns, scope):
    if len(bit_columns) != 11:
        return

    bits = frame_b1[bit_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    if bits.isna().any().any():
        record_error(scope, "B1: non-numeric or missing CAN-ID bits")
        return

    bit_values = bits.to_numpy()

    if not np.isin(bit_values, [0, 1]).all():
        record_error(scope, "B1: CAN-ID bit columns contain values other than 0/1")
        return

    # First stored bit is the most significant bit.
    weights = 2 ** np.arange(10, -1, -1)
    reconstructed_ids = (bit_values * weights).sum(axis=1)

    original_ids = pd.to_numeric(
        frame_e1["arbitration_id"], errors="coerce"
    ).to_numpy()

    if not np.array_equal(reconstructed_ids, original_ids):
        mismatch_count = int(
            np.count_nonzero(reconstructed_ids != original_ids)
        )

        record_error(
            scope,
            f"B1: CAN-ID reconstruction failed for "
            f"{mismatch_count} rows in this chunk"
        )


# ---------------------------------------------------------------------
# Partition validation
# ---------------------------------------------------------------------

def validate_partition(partition, expected_filenames=None):
    print(f"\nChecking partition: {partition}")

    partition_files = {}

    for representation in REPRESENTATIONS:
        directory = REPRESENTATIONS_DIR / representation / partition

        try:
            partition_files[representation] = get_csv_files(directory)
        except Exception as exc:
            record_error(partition, str(exc))
            return

    reference_files = partition_files["E1"]

    for representation in REPRESENTATIONS[1:]:
        if partition_files[representation] != reference_files:
            record_error(
                partition,
                f"{representation}: recording file set differs from E1"
            )

    if expected_filenames is not None:
        if reference_files != expected_filenames:
            missing = sorted(expected_filenames - reference_files)
            unexpected = sorted(reference_files - expected_filenames)

            record_error(
                partition,
                f"Recording coverage mismatch. Missing: {missing}; "
                f"unexpected: {unexpected}"
            )

    if not reference_files:
        record_error(partition, "No representation CSV files found")
        return

    for filename in sorted(reference_files):
        scope = f"{partition}/{filename}"

        paths = {
            representation:
                REPRESENTATIONS_DIR / representation / partition / filename
            for representation in REPRESENTATIONS
        }

        try:
            headers = {
                representation: get_headers(path)
                for representation, path in paths.items()
            }
        except Exception as exc:
            record_error(scope, f"Could not read CSV header: {exc}")
            continue

        for representation in REPRESENTATIONS:
            validate_header(
                representation,
                headers[representation],
                scope
            )

        if any(TARGET_COLUMN not in headers[r] for r in REPRESENTATIONS):
            continue

        # Identify B1 bit columns using the known non-bit columns.
        b1_non_bit_columns = [
            "payload_length",
            "byte_0",
            "byte_1",
            "byte_2",
            "byte_3",
            "byte_4",
            "byte_5",
            "byte_6",
            "byte_7",
            TARGET_COLUMN,
        ]

        b1_bit_columns = [
            column for column in headers["B1"]
            if column not in b1_non_bit_columns
        ]

        row_counts = {representation: 0 for representation in REPRESENTATIONS}
        chunk_iterators = {
            representation: pd.read_csv(
                paths[representation],
                chunksize=CHUNK_SIZE
            )
            for representation in REPRESENTATIONS
        }

        chunk_number = 0

        while True:
            chunks = {}

            try:
                for representation in REPRESENTATIONS:
                    chunks[representation] = next(
                        chunk_iterators[representation]
                    )
            except StopIteration:
                # All representations must finish at the same time.
                unfinished = []

                for representation, iterator in chunk_iterators.items():
                    try:
                        next(iterator)
                        unfinished.append(representation)
                    except StopIteration:
                        pass

                if unfinished:
                    record_error(
                        scope,
                        "Representations have different numbers of chunks; "
                        f"additional data found in {unfinished}"
                    )
                break

            chunk_number += 1

            for representation, frame in chunks.items():
                row_counts[representation] += len(frame)

                validate_numeric_chunk(
                    frame,
                    representation,
                    f"{scope}/chunk_{chunk_number}"
                )

                validate_labels(
                    frame[TARGET_COLUMN],
                    representation,
                    f"{scope}/chunk_{chunk_number}"
                )

            # Compare labels and row counts across all representations.
            reference_labels = chunks["E1"][TARGET_COLUMN].reset_index(drop=True)

            for representation in REPRESENTATIONS[1:]:
                current_labels = (
                    chunks[representation][TARGET_COLUMN]
                    .reset_index(drop=True)
                )

                if len(current_labels) != len(reference_labels):
                    record_error(
                        scope,
                        f"{representation}: chunk row count differs from E1"
                    )
                elif not current_labels.equals(reference_labels):
                    record_error(
                        scope,
                        f"{representation}: target labels/order differ "
                        f"from E1 in chunk {chunk_number}"
                    )

            # B1 must reconstruct the original integer CAN ID.
            if (
                len(b1_bit_columns) == 11
                and "arbitration_id" in chunks["E1"].columns
            ):
                validate_b1_id(
                    chunks["B1"],
                    chunks["E1"],
                    b1_bit_columns,
                    f"{scope}/chunk_{chunk_number}"
                )

        if len(set(row_counts.values())) != 1:
            record_error(
                scope,
                f"Total row counts differ across representations: "
                f"{row_counts}"
            )

        results.append({
            "partition": partition,
            "filename": filename,
            "rows": row_counts.get("E1", 0),
            "E1_rows": row_counts.get("E1", 0),
            "E4_rows": row_counts.get("E4", 0),
            "E5_rows": row_counts.get("E5", 0),
            "B1_rows": row_counts.get("B1", 0),
            "status": "PASS" if not any(
                error["scope"].startswith(scope)
                for error in errors
            ) else "FAIL",
        })

        print(
            f"  {filename}: "
            f"{row_counts.get('E1', 0):,} rows checked"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 65)
    print("CAN BUS REPRESENTATION VALIDATION")
    print("=" * 65)

    try:
        training_files = read_manifest_filenames(
            SPLIT_DIR / "training_files.csv"
        )
        validation_files = read_manifest_filenames(
            SPLIT_DIR / "validation_files.csv"
        )
    except Exception as exc:
        print(f"\nFATAL: Could not read split manifests: {exc}")
        return

    print(f"\nTraining recordings in manifest: {len(training_files)}")
    print(f"Validation recordings in manifest: {len(validation_files)}")

    if len(training_files) != 7:
        record_error(
            "split_manifest",
            f"Expected 7 training recordings, found {len(training_files)}"
        )

    if len(validation_files) != 5:
        record_error(
            "split_manifest",
            f"Expected 5 validation recordings, found {len(validation_files)}"
        )

    overlap = training_files & validation_files

    if overlap:
        record_error(
            "split_manifest",
            f"Training/validation overlap: {sorted(overlap)}"
        )

    validate_partition(TRAINING_PARTITION, training_files)
    validate_partition(VALIDATION_PARTITION, validation_files)

    for partition in TEST_PARTITIONS:
        validate_partition(partition)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_path = REPORT_DIR / "representation_validation_report.csv"
    results_df.to_csv(results_path, index=False)

    error_path = REPORT_DIR / "representation_validation_errors.csv"
    pd.DataFrame(
        errors,
        columns=["scope", "message"]
    ).to_csv(error_path, index=False)

    summary = {
        "training_recordings_in_manifest": len(training_files),
        "validation_recordings_in_manifest": len(validation_files),
        "training_validation_overlap": sorted(overlap),
        "recording_files_checked": len(results),
        "passed_recordings": sum(
            result["status"] == "PASS" for result in results
        ),
        "failed_recordings": sum(
            result["status"] == "FAIL" for result in results
        ),
        "error_count": len(errors),
        "status": "PASSED" if not errors else "FAILED",
    }

    summary_path = REPORT_DIR / "representation_validation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    print(f"Recording files checked: {summary['recording_files_checked']}")
    print(f"Passed recordings:       {summary['passed_recordings']}")
    print(f"Failed recordings:       {summary['failed_recordings']}")
    print(f"Errors:                  {summary['error_count']}")

    print(f"\nReport: {results_path}")
    print(f"Errors: {error_path}")
    print(f"Summary: {summary_path}")

    if errors:
        print("\nREPRESENTATION VALIDATION FAILED")
        print("Inspect representation_validation_errors.csv")
    else:
        print("\nREPRESENTATION VALIDATION PASSED")


if __name__ == "__main__":
    main()
