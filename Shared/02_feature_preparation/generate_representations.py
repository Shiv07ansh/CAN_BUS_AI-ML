
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

DATA_ROOT = (
    SHARED_ROOT
    / "data"
    / "processed"
    / "set_01"
)

ORIGINAL_DIR = DATA_ROOT / "features" / "original"
ENGINEERED_DIR = DATA_ROOT / "features" / "engineered"
SPLIT_DIR = DATA_ROOT / "split"
OUTPUT_ROOT = DATA_ROOT / "representations"

CHUNK_SIZE = 200_000

PARTITIONS = [
    "train_01",
    "test_01_known_vehicle_known_attack",
    "test_02_unknown_vehicle_known_attack",
    "test_03_known_vehicle_unknown_attack",
    "test_04_unknown_vehicle_unknown_attack",
]

BASE_FEATURES = [
    "arbitration_id",
    "payload_length",
    *[f"byte_{i}" for i in range(8)],
]

TARGET = "attack"

ALL_ENGINEERED_FEATURES = [
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

BINARY_ID_FEATURES = [
    f"id_bit_{bit}"
    for bit in range(10, -1, -1)
]

BYTE_FEATURES = [
    f"byte_{i}" for i in range(8)
]

# Each representation has a fixed, explicit feature order.
REPRESENTATIONS = {
    "E1": BASE_FEATURES,

    "E4": (
        BASE_FEATURES
        + COMPACT_ENGINEERED_FEATURES
    ),

    "E5": (
        BASE_FEATURES
        + ALL_ENGINEERED_FEATURES
    ),

    "B1": (
        BINARY_ID_FEATURES
        + ["payload_length"]
        + BYTE_FEATURES
    ),
}


# ============================================================
# SPLIT MANIFEST READING
# ============================================================

def read_manifest_filenames(manifest_path):
    """
    Extract CSV filenames from a split manifest.

    The function accepts manifests where filenames appear
    in any column, rather than assuming a particular column name.
    """

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        dtype=str,
        keep_default_na=False,
    )

    filenames = set()

    for column in manifest.columns:
        for value in manifest[column]:

            value = str(value).strip()

            if value.lower().endswith(".csv"):
                filenames.add(Path(value).name)

    if not filenames:
        raise ValueError(
            f"No CSV filenames found in {manifest_path}"
        )

    return filenames


def load_split_manifests():

    training_files = read_manifest_filenames(
        SPLIT_DIR / "training_files.csv"
    )

    validation_files = read_manifest_filenames(
        SPLIT_DIR / "validation_files.csv"
    )

    overlap = training_files & validation_files

    if overlap:
        raise ValueError(
            f"Training/validation overlap: {sorted(overlap)}"
        )

    return training_files, validation_files


# ============================================================
# BINARY CAN-ID ENCODING
# ============================================================

def encode_binary_can_id(can_ids):
    """
    Convert standard 11-bit CAN IDs into MSB-first binary columns.

    Example:
        CAN ID 0x123 -> 00100100011
    """

    can_ids = np.asarray(can_ids, dtype=np.int64)

    if np.any((can_ids < 0) | (can_ids > 0x7FF)):
        raise ValueError(
            "CAN ID outside the standard 11-bit range."
        )

    return pd.DataFrame({
        f"id_bit_{bit}": (
            (can_ids >> bit) & 1
        ).astype(np.int8)
        for bit in range(10, -1, -1)
    })


# ============================================================
# REPRESENTATION GENERATION
# ============================================================

def build_representation(name, original, engineered=None):

    if name == "E1":
        return original[BASE_FEATURES + [TARGET]].copy()

    if name == "E4":
        if engineered is None:
            raise ValueError("E4 requires engineered features.")

        result = pd.concat(
            [
                original[BASE_FEATURES].reset_index(drop=True),
                engineered[
                    COMPACT_ENGINEERED_FEATURES
                ].reset_index(drop=True),
                original[[TARGET]].reset_index(drop=True),
            ],
            axis=1,
        )

        return result

    if name == "E5":
        if engineered is None:
            raise ValueError("E5 requires engineered features.")

        result = pd.concat(
            [
                original[BASE_FEATURES].reset_index(drop=True),
                engineered[
                    ALL_ENGINEERED_FEATURES
                ].reset_index(drop=True),
                original[[TARGET]].reset_index(drop=True),
            ],
            axis=1,
        )

        return result

    if name == "B1":

        binary_ids = encode_binary_can_id(
            original["arbitration_id"].to_numpy()
        )

        result = pd.concat(
            [
                binary_ids,
                original[
                    ["payload_length"] + BYTE_FEATURES
                ].reset_index(drop=True),
                original[[TARGET]].reset_index(drop=True),
            ],
            axis=1,
        )

        return result

    raise ValueError(
        f"Unknown representation: {name}"
    )


# ============================================================
# RECORDING PROCESSING
# ============================================================

def process_recording(
    partition,
    filename,
    destination_group,
):

    original_path = (
        ORIGINAL_DIR / partition / filename
    )

    engineered_path = (
        ENGINEERED_DIR / partition / filename
    )

    if not original_path.exists():
        raise FileNotFoundError(original_path)

    # E4 and E5 need the engineered representation.
    if not engineered_path.exists():
        raise FileNotFoundError(engineered_path)

    output_paths = {}

    for representation_name in REPRESENTATIONS:

        output_dir = (
            OUTPUT_ROOT
            / representation_name
            / destination_group
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = output_dir / filename

        # Remove only the output file being regenerated.
        if output_path.exists():
            output_path.unlink()

        output_paths[representation_name] = output_path

    original_reader = pd.read_csv(
        original_path,
        chunksize=CHUNK_SIZE,
    )

    engineered_reader = pd.read_csv(
        engineered_path,
        chunksize=CHUNK_SIZE,
    )

    total_rows = 0

    for original_chunk, engineered_chunk in zip_longest(
        original_reader,
        engineered_reader,
    ):

        if original_chunk is None or engineered_chunk is None:
            raise ValueError(
                f"Chunk count mismatch for {filename}"
            )

        if len(original_chunk) != len(engineered_chunk):
            raise ValueError(
                f"Row count mismatch for {filename}"
            )

        if not np.array_equal(
            original_chunk["attack"].to_numpy(),
            engineered_chunk["attack"].to_numpy(),
        ):
            raise ValueError(
                f"Label mismatch for {filename}"
            )

        # Check original feature IDs and payload values
        # are aligned with their engineered counterparts.
        if not np.array_equal(
            original_chunk[BASE_FEATURES].to_numpy(),
            engineered_chunk[BASE_FEATURES].to_numpy(),
        ):
            raise ValueError(
                f"Original feature mismatch for {filename}"
            )

        for representation_name in REPRESENTATIONS:

            representation = build_representation(
                representation_name,
                original_chunk,
                engineered_chunk,
            )

            expected_columns = (
                REPRESENTATIONS[representation_name]
                + [TARGET]
            )

            if list(representation.columns) != expected_columns:
                raise ValueError(
                    f"Incorrect column order in "
                    f"{representation_name}/{filename}"
                )

            representation.to_csv(
                output_paths[representation_name],
                mode="a",
                header=(total_rows == 0),
                index=False,
            )

        total_rows += len(original_chunk)

    return total_rows


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(ORIGINAL_DIR)

    if not ENGINEERED_DIR.exists():
        raise FileNotFoundError(ENGINEERED_DIR)

    training_files, validation_files = load_split_manifests()

    # Verify the manifests against the actual train_01 inventory.
    train_inventory = {
        file.name
        for file in (ORIGINAL_DIR / "train_01").glob("*.csv")
    }

    if training_files & validation_files != set():
        raise ValueError("Split overlap detected.")

    if training_files | validation_files != train_inventory:
        raise ValueError(
            "Split manifests do not cover exactly the train_01 files.\n"
            f"Missing from manifests: "
            f"{sorted(train_inventory - training_files - validation_files)}\n"
            f"Unexpected manifest entries: "
            f"{sorted((training_files | validation_files) - train_inventory)}"
        )

    print("\n========== REPRESENTATION GENERATION ==========")
    print(f"Training recordings: {len(training_files)}")
    print(f"Validation recordings: {len(validation_files)}")
    print("Training/validation overlap: None")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    total_rows = {
        name: 0 for name in REPRESENTATIONS
    }

    total_files = 0

    # --------------------------------------------------------
    # Training and validation recordings.
    # --------------------------------------------------------

    for filename in sorted(train_inventory):

        if filename in training_files:
            destination_group = "training"
        else:
            destination_group = "validation"

        print(
            f"\n[{destination_group.upper()}] {filename}"
        )

        rows = process_recording(
            "train_01",
            filename,
            destination_group,
        )

        total_files += 1

        for name in REPRESENTATIONS:
            total_rows[name] += rows

        print(f"  Rows: {rows:,}")

    # --------------------------------------------------------
    # Official test partitions.
    # These remain separate and are never used for tuning.
    # --------------------------------------------------------

    for partition in PARTITIONS[1:]:

        print(f"\n--- Official test partition: {partition} ---")

        partition_dir = ORIGINAL_DIR / partition

        files = sorted(partition_dir.glob("*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No original feature files in {partition_dir}"
            )

        for input_path in files:

            filename = input_path.name

            print(f"Processing: {filename}")

            rows = process_recording(
                partition,
                filename,
                partition,
            )

            total_files += 1

            for name in REPRESENTATIONS:
                total_rows[name] += rows

            print(f"  Rows: {rows:,}")

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print("\n========== GENERATION SUMMARY ==========")
    print(f"Recordings processed: {total_files}")

    for name, features in REPRESENTATIONS.items():

        print(
            f"{name}: {len(features)} features + 1 target"
        )

        print(
            f"  Rows generated: {total_rows[name]:,}"
        )

        print(
            f"  Output: {OUTPUT_ROOT / name}"
        )

    print("\nGENERATION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
