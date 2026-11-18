
from pathlib import Path
import re

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT.parent / "data" / "sampled_stratified" / "set_01"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "set_01"

CHUNK_SIZE = 200_000

EXPECTED_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]

PARTITIONS = [
    "train_01",
    "test_01_known_vehicle_known_attack",
    "test_02_unknown_vehicle_known_attack",
    "test_03_known_vehicle_unknown_attack",
    "test_04_unknown_vehicle_unknown_attack",
]

PAYLOAD_PATTERN = re.compile(r"(?:[0-9a-fA-F]{2}){0,8}")


# ============================================================
# VALIDATION HELPERS
# ============================================================

def valid_can_id(value):
    try:
        can_id = int(str(value).strip(), 16)
        return 0 <= can_id <= 0x7FF
    except (ValueError, TypeError):
        return False


def valid_payload(value):
    value = str(value).strip()

    # Empty payloads are valid in this dataset.
    return bool(PAYLOAD_PATTERN.fullmatch(value))


def validate_file(file_path):
    print(f"Validating: {file_path.name}")

    stats = {
        "partition": file_path.parent.name,
        "file": file_path.name,
        "rows": 0,
        "normal_rows": 0,
        "attack_rows": 0,
        "blank_timestamps": 0,
        "blank_can_ids": 0,
        "blank_payloads": 0,
        "invalid_can_ids": 0,
        "invalid_payloads": 0,
        "invalid_labels": 0,
    }

    header = pd.read_csv(file_path, nrows=0).columns.tolist()

    if header != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {file_path}.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Found: {header}"
        )

    for chunk in pd.read_csv(
        file_path,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        stats["rows"] += len(chunk)

        timestamps = chunk["timestamp"].str.strip()
        can_ids = chunk["arbitration_id"].str.strip()
        payloads = chunk["data_field"].str.strip()
        labels = chunk["attack"].str.strip()

        stats["blank_timestamps"] += timestamps.eq("").sum()
        stats["blank_can_ids"] += can_ids.eq("").sum()
        stats["blank_payloads"] += payloads.eq("").sum()

        stats["normal_rows"] += labels.eq("0").sum()
        stats["attack_rows"] += labels.eq("1").sum()

        stats["invalid_labels"] += (
            ~labels.isin(["0", "1"])
        ).sum()

        stats["invalid_can_ids"] += (
            ~can_ids.map(valid_can_id)
        ).sum()

        stats["invalid_payloads"] += (
            ~payloads.map(valid_payload)
        ).sum()

    return stats


# ============================================================
# MAIN
# ============================================================

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    all_stats = []

    for partition in PARTITIONS:
        partition_dir = DATASET_DIR / partition

        if not partition_dir.exists():
            raise FileNotFoundError(
                f"Missing partition: {partition_dir}"
            )

        csv_files = sorted(partition_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {partition_dir}"
            )

        for file_path in csv_files:
            all_stats.append(validate_file(file_path))

    audit = pd.DataFrame(all_stats)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "dataset_audit.csv"
    audit.to_csv(output_path, index=False)

    print("\n========== DATASET VALIDATION ==========")
    print(f"Files checked: {len(audit)}")
    print(f"Total rows: {audit['rows'].sum():,}")
    print(f"Normal rows: {audit['normal_rows'].sum():,}")
    print(f"Attack rows: {audit['attack_rows'].sum():,}")

    error_columns = [
        "blank_timestamps",
        "blank_can_ids",
        "invalid_can_ids",
        "invalid_payloads",
        "invalid_labels",
    ]

    errors = audit[error_columns].sum()

    print("\nValidation findings:")
    print(errors.to_string())

    print(
        f"\nBlank payloads: "
        f"{audit['blank_payloads'].sum():,} "
        "(permitted)"
    )

    if errors.sum() > 0:
        raise ValueError(
            "Dataset validation failed. "
            "Inspect dataset_audit.csv."
        )

    print("\nVALIDATION PASSED")
    print(f"Audit saved to: {output_path}")


if __name__ == "__main__":
    main()
