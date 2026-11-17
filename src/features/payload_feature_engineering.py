
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "prepared" / "set_01" / "train_01"
OUTPUT_DIR = ROOT / "data" / "engineered" / "set_01" / "train_01"
REPORT_DIR = ROOT / "reports" / "set_01" / "stage_6" / "payload_feature_engineering"

BYTE_COLUMNS = [f"byte_{i}" for i in range(8)]

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


def calculate_payload_features(df):
    """Calculate message-level statistics using only valid payload bytes."""

    byte_values = df[BYTE_COLUMNS].to_numpy(dtype=np.int16)

    # -1 represents a missing byte, not an actual payload value.
    valid = byte_values >= 0
    lengths = valid.sum(axis=1)

    safe_values = np.where(valid, byte_values, 0)

    zero_count = ((safe_values == 0) & valid).sum(axis=1)
    nonzero_count = ((safe_values != 0) & valid).sum(axis=1)

    sums = safe_values.sum(axis=1, dtype=np.float64)

    means = np.divide(
        sums,
        lengths,
        out=np.zeros(len(df), dtype=np.float64),
        where=lengths > 0,
    )

    squared_differences = np.where(
        valid,
        (safe_values - means[:, None]) ** 2,
        0.0,
    )

    variances = np.divide(
        squared_differences.sum(axis=1),
        lengths,
        out=np.zeros(len(df), dtype=np.float64),
        where=lengths > 0,
    )
    standard_deviations = np.sqrt(variances)

    # Calculate Shannon entropy of the byte-value distribution.
    # For each valid byte, count how often its value occurs in that message.
    entropy = np.zeros(len(df), dtype=np.float64)

    for column_index in range(8):
        current_values = safe_values[:, column_index]

        matches = (
            (safe_values == current_values[:, None])
            & valid
            & valid[:, column_index, None]
        )

        occurrence_counts = matches.sum(axis=1)

        probabilities = np.divide(
            occurrence_counts,
            lengths,
            out=np.ones(len(df), dtype=np.float64),
            where=lengths > 0,
        )

        contribution = np.zeros(len(df), dtype=np.float64)
        usable = valid[:, column_index] & (lengths > 0)

        contribution[usable] = (
            -np.log2(probabilities[usable]) / lengths[usable]
        )

        entropy += contribution

    unique_byte_count = np.zeros(len(df), dtype=np.int8)

    # At most eight bytes per message, so this small loop is inexpensive.
    for column_index in range(8):
        is_first_occurrence = valid[:, column_index].copy()

        for previous_index in range(column_index):
            is_first_occurrence &= (
                safe_values[:, column_index]
                != safe_values[:, previous_index]
            ) | ~valid[:, previous_index]

        unique_byte_count += is_first_occurrence.astype(np.int8)

    # Lookup table for the number of set bits in each possible byte value.
    bit_count_lookup = np.array(
        [bin(value).count("1") for value in range(256)],
        dtype=np.int8,
    )

    bit_ones_count = bit_count_lookup[safe_values].astype(np.int16)
    bit_ones_count = np.where(valid, bit_ones_count, 0).sum(axis=1)

    total_valid_bits = lengths * 8

    bit_density = np.divide(
        bit_ones_count,
        total_valid_bits,
        out=np.zeros(len(df), dtype=np.float64),
        where=total_valid_bits > 0,
    )

    result = df.copy()

    result["payload_zero_count"] = zero_count.astype(np.int8)
    result["payload_nonzero_count"] = nonzero_count.astype(np.int8)
    result["payload_byte_mean"] = means
    result["payload_byte_std"] = standard_deviations
    result["payload_byte_entropy"] = entropy
    result["payload_unique_byte_count"] = unique_byte_count
    result["payload_bit_ones_count"] = bit_ones_count.astype(np.int8)
    result["payload_bit_density"] = bit_density

    return result


def validate_engineered_features(original, engineered, file_name):
    assert len(original) == len(engineered), (
        f"{file_name}: row count changed"
    )

    assert original["attack"].equals(engineered["attack"]), (
        f"{file_name}: target labels changed"
    )

    for column in BYTE_COLUMNS + ["arbitration_id", "payload_length"]:
        assert original[column].equals(engineered[column]), (
            f"{file_name}: original feature {column} changed"
        )

    for column in ENGINEERED_COLUMNS:
        assert column in engineered.columns
        assert np.isfinite(engineered[column].to_numpy()).all(), (
            f"{file_name}: non-finite values in {column}"
        )

    lengths = engineered["payload_length"].to_numpy()

    assert (
        (engineered["payload_zero_count"].to_numpy()
        + engineered["payload_nonzero_count"].to_numpy()
        == lengths).all()
    ), f"{file_name}: zero/nonzero counts do not match payload length"

    assert (
        engineered["payload_unique_byte_count"].to_numpy()
        <= lengths
    ).all(), f"{file_name}: unique-byte count exceeds payload length"

    assert (
        engineered["payload_bit_ones_count"].to_numpy()
        <= lengths * 8
    ).all(), f"{file_name}: bit count exceeds available payload bits"

    assert (
        (engineered["payload_bit_density"] >= 0)
        & (engineered["payload_bit_density"] <= 1)
    ).all(), f"{file_name}: bit density outside [0, 1]"

    empty_rows = lengths == 0

    if empty_rows.any():
        for column in ENGINEERED_COLUMNS:
            assert (
                engineered.loc[empty_rows, column] == 0
            ).all(), f"{file_name}: empty payload has nonzero {column}"


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    input_files = sorted(INPUT_DIR.rglob("*.csv"))

    if not input_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "files_processed": [],
        "total_rows": 0,
        "engineered_features": ENGINEERED_COLUMNS,
        "validation": "passed",
    }

    print(f"Found {len(input_files)} CSV files.")
    print("Generating payload-level features...\n")

    for input_file in input_files:
        print(f"Processing: {input_file.name}")

        original = pd.read_csv(input_file)

        required_columns = (
            BYTE_COLUMNS
            + ["arbitration_id", "payload_length", "attack"]
        )

        missing = [
            column for column in required_columns
            if column not in original.columns
        ]

        if missing:
            raise ValueError(
                f"{input_file.name}: missing columns {missing}"
            )

        engineered = calculate_payload_features(original)

        validate_engineered_features(
            original,
            engineered,
            input_file.name,
        )

        relative_path = input_file.relative_to(INPUT_DIR)
        output_file = OUTPUT_DIR / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        engineered.to_csv(output_file, index=False)

        metadata["files_processed"].append({
            "file": str(relative_path),
            "rows": int(len(engineered)),
            "attack_rows": int(engineered["attack"].sum()),
            "output": str(output_file),
            "validation": "passed",
        })

        metadata["total_rows"] += int(len(engineered))

        print(
            f"  Rows: {len(engineered):,} | "
            f"Attacks: {int(engineered['attack'].sum()):,} | "
            "Validation: PASS"
        )

    metadata_path = REPORT_DIR / "payload_feature_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n========================================")
    print("PAYLOAD FEATURE ENGINEERING COMPLETE")
    print("========================================")
    print(f"Files processed: {len(input_files)}")
    print(f"Total rows: {metadata['total_rows']:,}")
    print(f"New features: {len(ENGINEERED_COLUMNS)}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Metadata: {metadata_path}")
    print("All file-level validations passed.")


if __name__ == "__main__":
    main()
