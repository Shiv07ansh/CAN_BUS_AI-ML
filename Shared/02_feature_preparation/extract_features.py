
from pathlib import Path
import re
import sys

import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = SHARED_ROOT.parent

RAW_DATA_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "sampled_stratified"
    / "set_01"
)

OUTPUT_DIR = (
    SHARED_ROOT
    / "data"
    / "processed"
    / "set_01"
    / "features"
    / "original"
)

CHUNK_SIZE = 200_000

PARTITIONS = [
    "train_01",
    "test_01_known_vehicle_known_attack",
    "test_02_unknown_vehicle_known_attack",
    "test_03_known_vehicle_unknown_attack",
    "test_04_unknown_vehicle_unknown_attack",
]

EXPECTED_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]

FEATURE_COLUMNS = [
    "arbitration_id",
    "payload_length",
    *[f"byte_{i}" for i in range(8)],
    "attack",
]

PAYLOAD_PATTERN = re.compile(r"(?:[0-9a-fA-F]{2}){0,8}")


# ============================================================
# CONVERSION FUNCTIONS
# ============================================================

def parse_can_id(value):
    """Convert a hexadecimal CAN arbitration ID to an integer."""

    value = str(value).strip()

    if not value:
        raise ValueError("Blank arbitration ID encountered.")

    if value.lower().startswith("0x"):
        value = value[2:]

    can_id = int(value, 16)

    if not 0 <= can_id <= 0x7FF:
        raise ValueError(f"CAN ID outside standard 11-bit range: {value}")

    return can_id


def parse_payload(value):
    """
    Convert a hexadecimal CAN payload into eight byte features.

    Missing byte positions are represented by -1.
    Empty payloads are valid zero-DLC messages.
    """

    value = str(value).strip()

    if not PAYLOAD_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid CAN payload: {value!r}")

    payload_bytes = [
        int(value[i:i + 2], 16)
        for i in range(0, len(value), 2)
    ]

    payload_length = len(payload_bytes)

    padded_bytes = payload_bytes + [-1] * (8 - payload_length)

    return [payload_length, *padded_bytes]


# ============================================================
# FILE PROCESSING
# ============================================================

def extract_file(input_path, output_path):

    print(f"Processing: {input_path.name}")

    if output_path.exists():
        output_path.unlink()

    total_rows = 0

    for chunk in pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        if list(chunk.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {input_path}.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found: {list(chunk.columns)}"
            )

        # Convert arbitration IDs.
        chunk["arbitration_id"] = (
            chunk["arbitration_id"].map(parse_can_id)
        )

        # Validate and convert labels.
        labels = chunk["attack"].astype(str).str.strip()

        if not labels.isin(["0", "1"]).all():
            raise ValueError(
                f"Invalid attack labels found in {input_path}"
            )

        chunk["attack"] = labels.astype("int8")

        # Extract payload features.
        payload_features = chunk["data_field"].map(parse_payload)

        payload_df = pd.DataFrame(
            payload_features.tolist(),
            columns=[
                "payload_length",
                *[f"byte_{i}" for i in range(8)],
            ],
            index=chunk.index,
        )

        # Assemble the final feature matrix.
        output_chunk = pd.concat(
            [
                chunk[["arbitration_id"]],
                payload_df,
                chunk[["attack"]],
            ],
            axis=1,
        )

        output_chunk = output_chunk[FEATURE_COLUMNS]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_chunk.to_csv(
            output_path,
            mode="a",
            header=(total_rows == 0),
            index=False,
        )

        total_rows += len(output_chunk)

    print(f"  Rows extracted: {total_rows:,}")
    print(f"  Saved to: {output_path}")

    return total_rows


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DATA_DIR}"
        )

    total_files = 0
    total_rows = 0

    print("\n========== FEATURE EXTRACTION ==========")
    print(f"Input:  {RAW_DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    for partition in PARTITIONS:

        partition_dir = RAW_DATA_DIR / partition

        if not partition_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset partition: {partition_dir}"
            )

        files = sorted(partition_dir.glob("*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in {partition_dir}"
            )

        print(f"\n--- Partition: {partition} ---")

        for input_path in files:

            output_path = (
                OUTPUT_DIR
                / partition
                / input_path.name
            )

            rows = extract_file(input_path, output_path)

            total_files += 1
            total_rows += rows

    print("\n========== EXTRACTION COMPLETE ==========")
    print(f"Files processed: {total_files}")
    print(f"Total rows: {total_rows:,}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
