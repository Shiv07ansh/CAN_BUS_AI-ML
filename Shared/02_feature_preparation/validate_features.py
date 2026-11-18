
from pathlib import Path
from itertools import zip_longest
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_ROOT = SCRIPT_DIR.parent

FEATURE_ROOT = (
    SHARED_ROOT
    / "data"
    / "processed"
    / "set_01"
    / "features"
)

ORIGINAL_DIR = FEATURE_ROOT / "original"
ENGINEERED_DIR = FEATURE_ROOT / "engineered"

REPORT_DIR = (
    SHARED_ROOT
    / "data"
    / "processed"
    / "set_01"
    / "validation"
)

REPORT_PATH = REPORT_DIR / "feature_validation_report.csv"

CHUNK_SIZE = 200_000

PARTITIONS = [
    "train_01",
    "test_01_known_vehicle_known_attack",
    "test_02_unknown_vehicle_known_attack",
    "test_03_known_vehicle_unknown_attack",
    "test_04_unknown_vehicle_unknown_attack",
]

BASE_COLUMNS = [
    "arbitration_id",
    "payload_length",
    *[f"byte_{i}" for i in range(8)],
    "attack",
]

ENGINEERED_COLUMNS = [
    "payload_zero_count",
    "payload_nonzero_count",
    "payload_byte_mean",
    "payload_byte_std",
    "payload_byte_entropy",
    "payload_unique_byte_count",
    "payload_bit_ones_count",
    "payload_bit_density",
]

EXPECTED_ENGINEERED_COLUMNS = BASE_COLUMNS + ENGINEERED_COLUMNS

BYTE_COLUMNS = [f"byte_{i}" for i in range(8)]

FLOAT_FEATURES = [
    "payload_byte_mean",
    "payload_byte_std",
    "payload_byte_entropy",
    "payload_bit_density",
]

BIT_COUNT_LOOKUP = np.array(
    [bin(value).count("1") for value in range(256)],
    dtype=np.int16,
)


# ============================================================
# INDEPENDENT FEATURE RECALCULATION
# ============================================================

def recalculate_features(byte_values):
    """
    Independently recalculate all eight payload features.

    Missing byte positions are represented by -1.
    Empty payloads produce zero for all eight features.
    """

    valid_mask = byte_values >= 0
    valid_counts = valid_mask.sum(axis=1)

    safe_values = np.where(valid_mask, byte_values, 0)

    # Zero and non-zero counts.
    zero_count = (
        (valid_mask & (byte_values == 0)).sum(axis=1)
    )

    nonzero_count = (
        (valid_mask & (byte_values != 0)).sum(axis=1)
    )

    # Mean.
    byte_sum = safe_values.sum(axis=1, dtype=np.float64)

    byte_mean = np.divide(
        byte_sum,
        valid_counts,
        out=np.zeros(len(byte_values), dtype=np.float64),
        where=valid_counts > 0,
    )

    # Population standard deviation.
    deviations = np.where(
        valid_mask,
        byte_values - byte_mean[:, None],
        0.0,
    )

    byte_std = np.sqrt(
        np.divide(
            (deviations ** 2).sum(axis=1),
            valid_counts,
            out=np.zeros(len(byte_values), dtype=np.float64),
            where=valid_counts > 0,
        )
    )

        # Shannon entropy in bits.
    equality = (
        byte_values[:, :, None]
        == byte_values[:, None, :]
    )

    equality &= valid_mask[:, None, :]

    frequencies = equality.sum(axis=2)

    probabilities = np.divide(
        frequencies,
        valid_counts[:, None],
        out=np.ones_like(frequencies, dtype=np.float64),
        where=valid_counts[:, None] > 0,
    )

    entropy_terms = np.zeros_like(
        probabilities,
        dtype=np.float64,
    )

    mask = valid_mask & (frequencies > 0)

    entropy_terms[mask] = -np.log2(probabilities[mask])

    entropy = np.divide(
        entropy_terms.sum(axis=1),
        valid_counts,
        out=np.zeros(len(byte_values), dtype=np.float64),
        where=valid_counts > 0,
    )

    # Unique byte values.
    sorted_values = np.sort(
        np.where(valid_mask, byte_values, 256),
        axis=1,
    )

    first_valid = (
        (sorted_values[:, 0] < 256).astype(np.int16)
    )

    additional_unique = (
        (sorted_values[:, 1:] < 256)
        & (
            sorted_values[:, 1:]
            != sorted_values[:, :-1]
        )
    ).sum(axis=1)

    unique_count = first_valid + additional_unique

    # Number of set bits.
    bit_ones = BIT_COUNT_LOOKUP[safe_values].sum(axis=1)

    # Bit density.
    total_bits = valid_counts * 8

    bit_density = np.divide(
        bit_ones,
        total_bits,
        out=np.zeros(len(byte_values), dtype=np.float64),
        where=total_bits > 0,
    )

    return {
        "payload_zero_count": zero_count,
        "payload_nonzero_count": nonzero_count,
        "payload_byte_mean": byte_mean,
        "payload_byte_std": byte_std,
        "payload_byte_entropy": entropy,
        "payload_unique_byte_count": unique_count,
        "payload_bit_ones_count": bit_ones,
        "payload_bit_density": bit_density,
    }


# ============================================================
# CHUNK VALIDATION
# ============================================================

def validate_chunk(original, engineered, file_name):

    errors = []

    if list(original.columns) != BASE_COLUMNS:
        raise ValueError(
            f"Incorrect original schema in {file_name}"
        )

    if list(engineered.columns) != EXPECTED_ENGINEERED_COLUMNS:
        raise ValueError(
            f"Incorrect engineered schema in {file_name}"
        )

    if len(original) != len(engineered):
        raise ValueError(
            f"Row count mismatch in {file_name}"
        )

    # --------------------------------------------------------
    # Check original data preservation.
    # --------------------------------------------------------

    for column in BASE_COLUMNS:

        original_values = original[column].to_numpy()
        engineered_values = engineered[column].to_numpy()

        if not np.array_equal(
            original_values,
            engineered_values,
        ):
            errors.append(f"original_data_changed:{column}")

    # --------------------------------------------------------
    # Check missing and non-finite values.
    # --------------------------------------------------------

    if original.isna().any().any():
        errors.append("missing_values_in_original")

    if engineered.isna().any().any():
        errors.append("missing_values_in_engineered")

    numeric_engineered = engineered.select_dtypes(
        include=[np.number]
    )

    if not np.isfinite(
        numeric_engineered.to_numpy(dtype=np.float64)
    ).all():
        errors.append("non_finite_values")

    # --------------------------------------------------------
    # Check payload representation.
    # --------------------------------------------------------

    byte_values = engineered[BYTE_COLUMNS].to_numpy(
        dtype=np.int16
    )

    payload_lengths = engineered[
        "payload_length"
    ].to_numpy(dtype=np.int16)

    valid_mask = byte_values >= 0
    valid_counts = valid_mask.sum(axis=1)

    if np.any((byte_values < -1) | (byte_values > 255)):
        errors.append("byte_out_of_range")

    if np.any((payload_lengths < 0) | (payload_lengths > 8)):
        errors.append("payload_length_out_of_range")

    if not np.array_equal(valid_counts, payload_lengths):
        errors.append("payload_length_mismatch")

    # Check that valid bytes occupy positions from left to right.
    expected_mask = (
        np.arange(8)[None, :]
        < payload_lengths[:, None]
    )

    if not np.array_equal(valid_mask, expected_mask):
        errors.append("invalid_byte_padding")

    # --------------------------------------------------------
    # Check labels.
    # --------------------------------------------------------

    labels = engineered["attack"].to_numpy()

    if not np.isin(labels, [0, 1]).all():
        errors.append("invalid_attack_labels")

    # --------------------------------------------------------
    # Recalculate engineered features independently.
    # --------------------------------------------------------

    expected_features = recalculate_features(byte_values)

    for feature_name, expected_values in expected_features.items():

        actual_values = engineered[
            feature_name
        ].to_numpy(dtype=np.float64)

        if not np.allclose(
            actual_values,
            expected_values,
            rtol=1e-9,
            atol=1e-9,
        ):
            errors.append(
                f"incorrect_calculation:{feature_name}"
            )

    # --------------------------------------------------------
    # Check mathematical invariants.
    # --------------------------------------------------------

    zero_count = engineered[
        "payload_zero_count"
    ].to_numpy()

    nonzero_count = engineered[
        "payload_nonzero_count"
    ].to_numpy()

    unique_count = engineered[
        "payload_unique_byte_count"
    ].to_numpy()

    bit_ones = engineered[
        "payload_bit_ones_count"
    ].to_numpy()

    bit_density = engineered[
        "payload_bit_density"
    ].to_numpy()

    entropy = engineered[
        "payload_byte_entropy"
    ].to_numpy()

    byte_std = engineered[
        "payload_byte_std"
    ].to_numpy()

    if not np.array_equal(
        zero_count + nonzero_count,
        payload_lengths,
    ):
        errors.append("zero_nonzero_count_mismatch")

    if np.any(unique_count > payload_lengths):
        errors.append("unique_count_exceeds_length")

    if np.any(bit_ones > payload_lengths * 8):
        errors.append("bit_count_exceeds_payload_capacity")

    if np.any((bit_density < 0) | (bit_density > 1)):
        errors.append("bit_density_out_of_range")

    if np.any(entropy < -1e-12):
        errors.append("negative_entropy")

    if np.any(byte_std < -1e-12):
        errors.append("negative_standard_deviation")

    # --------------------------------------------------------
    # Check zero-DLC messages.
    # --------------------------------------------------------

    empty_mask = payload_lengths == 0

    if empty_mask.any():

        for feature_name in ENGINEERED_COLUMNS:

            values = engineered.loc[
                empty_mask,
                feature_name,
            ].to_numpy(dtype=np.float64)

            if not np.all(values == 0):
                errors.append(
                    f"empty_payload_not_zero:{feature_name}"
                )

    return errors


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_file(original_path, engineered_path):

    print(f"Validating: {original_path.name}")

    original_reader = pd.read_csv(
        original_path,
        chunksize=CHUNK_SIZE,
    )

    engineered_reader = pd.read_csv(
        engineered_path,
        chunksize=CHUNK_SIZE,
    )

    total_rows = 0
    error_counts = {}

    for original_chunk, engineered_chunk in zip_longest(
        original_reader,
        engineered_reader,
    ):

        if original_chunk is None or engineered_chunk is None:
            error_counts["chunk_count_mismatch"] = 1
            break

        errors = validate_chunk(
            original_chunk,
            engineered_chunk,
            original_path.name,
        )

        for error in errors:
            error_counts[error] = (
                error_counts.get(error, 0) + 1
            )

        total_rows += len(original_chunk)

    return {
        "file": original_path.name,
        "rows": total_rows,
        "status": "PASS" if not error_counts else "FAIL",
        "errors": ";".join(sorted(error_counts)),
    }


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(ORIGINAL_DIR)

    if not ENGINEERED_DIR.exists():
        raise FileNotFoundError(ENGINEERED_DIR)

    results = []

    total_files = 0
    total_rows = 0
    failed_files = 0

    print("\n========== FEATURE VALIDATION ==========")
    print(f"Original:   {ORIGINAL_DIR}")
    print(f"Engineered: {ENGINEERED_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    for partition in PARTITIONS:

        original_partition = ORIGINAL_DIR / partition
        engineered_partition = ENGINEERED_DIR / partition

        if not original_partition.exists():
            raise FileNotFoundError(original_partition)

        if not engineered_partition.exists():
            raise FileNotFoundError(engineered_partition)

        original_files = {
            file.name: file
            for file in original_partition.glob("*.csv")
        }

        engineered_files = {
            file.name: file
            for file in engineered_partition.glob("*.csv")
        }

        if set(original_files) != set(engineered_files):
            raise ValueError(
                f"File inventory mismatch in partition {partition}.\n"
                f"Original only: "
                f"{sorted(set(original_files) - set(engineered_files))}\n"
                f"Engineered only: "
                f"{sorted(set(engineered_files) - set(original_files))}"
            )

        print(f"\n--- Partition: {partition} ---")

        for file_name in sorted(original_files):

            result = validate_file(
                original_files[file_name],
                engineered_files[file_name],
            )

            result["partition"] = partition

            results.append(result)

            total_files += 1
            total_rows += result["rows"]

            if result["status"] == "FAIL":
                failed_files += 1
                print(f"  FAIL: {file_name}: {result['errors']}")
            else:
                print(f"  PASS: {file_name} ({result['rows']:,} rows)")

    # --------------------------------------------------------
    # Save validation report.
    # --------------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(results)

    report = report[
        ["partition", "file", "rows", "status", "errors"]
    ]

    report.to_csv(REPORT_PATH, index=False)

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"Files checked: {total_files}")
    print(f"Total rows checked: {total_rows:,}")
    print(f"Passed files: {total_files - failed_files}")
    print(f"Failed files: {failed_files}")
    print(f"Report: {REPORT_PATH}")

    if failed_files:
        print("\nVALIDATION FAILED")
        sys.exit(1)

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
