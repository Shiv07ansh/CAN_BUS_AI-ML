
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_ROOT = SCRIPT_DIR.parent

FEATURES_DIR = (
    SHARED_ROOT
    / "data"
    / "processed"
    / "set_01"
    / "features"
    / "original"
)

OUTPUT_DIR = (
    SHARED_ROOT
    / "data"
    / "processed"
    / "set_01"
    / "features"
    / "engineered"
)

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

OUTPUT_COLUMNS = BASE_COLUMNS + ENGINEERED_COLUMNS

# Lookup table for the number of set bits in each possible byte.
BIT_COUNT_LOOKUP = np.array(
    [bin(value).count("1") for value in range(256)],
    dtype=np.int16,
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_payload_features(byte_values):
    """
    Calculate eight payload statistics for each CAN message.

    byte_values:
        NumPy array with shape (n_rows, 8).

    Missing byte positions:
        -1

    Empty payload:
        All eight engineered features equal zero.
    """

    valid_mask = byte_values >= 0

    valid_counts = valid_mask.sum(axis=1).astype(np.int16)

    # Replace missing values with zero for safe arithmetic.
    # The validity mask ensures they are excluded from statistics.
    safe_values = np.where(valid_mask, byte_values, 0)

    # --------------------------------------------------------
    # 1. Zero-valued byte count
    # --------------------------------------------------------

    zero_count = (
        (valid_mask & (byte_values == 0))
        .sum(axis=1)
        .astype(np.int16)
    )

    # --------------------------------------------------------
    # 2. Non-zero byte count
    # --------------------------------------------------------

    nonzero_count = (
        (valid_mask & (byte_values != 0))
        .sum(axis=1)
        .astype(np.int16)
    )

    # --------------------------------------------------------
    # 3. Byte mean
    # --------------------------------------------------------

    byte_sum = safe_values.sum(axis=1, dtype=np.float64)

    byte_mean = np.divide(
        byte_sum,
        valid_counts,
        out=np.zeros(len(byte_values), dtype=np.float64),
        where=valid_counts > 0,
    )

    # --------------------------------------------------------
    # 4. Population standard deviation (ddof=0)
    # --------------------------------------------------------

    squared_sum = (
        safe_values.astype(np.float64) ** 2
    ).sum(axis=1)

    mean_squared = np.divide(
        squared_sum,
        valid_counts,
        out=np.zeros(len(byte_values), dtype=np.float64),
        where=valid_counts > 0,
    )

    byte_variance = np.maximum(
        mean_squared - byte_mean ** 2,
        0.0,
    )

    byte_std = np.sqrt(byte_variance)

    # --------------------------------------------------------
    # 5. Shannon entropy in bits
    # --------------------------------------------------------

    # Compare the eight byte positions against one another.
    # This produces the frequency of each byte value within
    # each individual CAN message.
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

    valid_positions = valid_mask & (frequencies > 0)

    entropy_terms[valid_positions] = (
        -np.log2(probabilities[valid_positions])
    )

    entropy = np.divide(
        entropy_terms.sum(axis=1),
        valid_counts,
        out=np.zeros(len(byte_values), dtype=np.float64),
        where=valid_counts > 0,
    )

    # --------------------------------------------------------
    # 6. Unique byte-value count
    # --------------------------------------------------------

    unique_count = (
        (valid_mask[:, :, None] & equality)
        .sum(axis=2)
    )

    # Each byte position contributes one if it is the first
    # occurrence of its value in the message.
    first_occurrence = (
        valid_mask
        & (
            np.tril(
                np.ones((8, 8), dtype=bool),
                k=-1,
            )[None, :, :]
            == False
        ).all(axis=2)
    )

    # A clearer pairwise formulation:
    earlier_equal = equality & np.triu(
        np.ones((8, 8), dtype=bool),
        k=1,
    )[None, :, :]

    is_duplicate = earlier_equal.any(axis=2)

    unique_count = (
        (valid_mask & ~is_duplicate)
        .sum(axis=1)
        .astype(np.int16)
    )

    # --------------------------------------------------------
    # 7. Total number of set bits
    # --------------------------------------------------------

    bit_ones = BIT_COUNT_LOOKUP[safe_values].sum(
        axis=1
    ).astype(np.int16)

    # --------------------------------------------------------
    # 8. Bit density
    # --------------------------------------------------------

    total_valid_bits = valid_counts * 8

    bit_density = np.divide(
        bit_ones,
        total_valid_bits,
        out=np.zeros(len(byte_values), dtype=np.float64),
        where=total_valid_bits > 0,
    )

    return pd.DataFrame({
        "payload_zero_count": zero_count,
        "payload_nonzero_count": nonzero_count,
        "payload_byte_mean": byte_mean,
        "payload_byte_std": byte_std,
        "payload_byte_entropy": entropy,
        "payload_unique_byte_count": unique_count,
        "payload_bit_ones_count": bit_ones,
        "payload_bit_density": bit_density,
    })


# ============================================================
# FILE PROCESSING
# ============================================================

def engineer_file(input_path, output_path):

    print(f"Processing: {input_path.name}")

    if output_path.exists():
        output_path.unlink()

    total_rows = 0

    for chunk in pd.read_csv(
        input_path,
        chunksize=CHUNK_SIZE,
    ):

        if list(chunk.columns) != BASE_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {input_path}.\n"
                f"Expected: {BASE_COLUMNS}\n"
                f"Found: {list(chunk.columns)}"
            )

        byte_columns = [
            f"byte_{i}" for i in range(8)
        ]

        byte_values = chunk[byte_columns].to_numpy(
            dtype=np.int16
        )

        payload_lengths = chunk[
            "payload_length"
        ].to_numpy(dtype=np.int16)

        # ----------------------------------------------------
        # Validate payload representation
        # ----------------------------------------------------

        if np.any((byte_values < -1) | (byte_values > 255)):
            raise ValueError(
                f"Invalid byte value in {input_path}"
            )

        if np.any((payload_lengths < 0) | (payload_lengths > 8)):
            raise ValueError(
                f"Invalid payload length in {input_path}"
            )

        valid_counts = (byte_values >= 0).sum(axis=1)

        if not np.array_equal(valid_counts, payload_lengths):
            raise ValueError(
                f"Payload length and byte representation mismatch "
                f"in {input_path}"
            )

        # ----------------------------------------------------
        # Calculate engineered features
        # ----------------------------------------------------

        engineered = calculate_payload_features(byte_values)

        output_chunk = pd.concat(
            [
                chunk.reset_index(drop=True),
                engineered,
            ],
            axis=1,
        )

        output_chunk = output_chunk[OUTPUT_COLUMNS]

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

    print(f"  Rows processed: {total_rows:,}")
    print(f"  Saved to: {output_path}")

    return total_rows


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    if not FEATURES_DIR.exists():
        raise FileNotFoundError(
            f"Original features directory not found: {FEATURES_DIR}"
        )

    total_files = 0
    total_rows = 0

    print("\n========== PAYLOAD FEATURE ENGINEERING ==========")
    print(f"Input:  {FEATURES_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    for partition in PARTITIONS:

        partition_dir = FEATURES_DIR / partition

        if not partition_dir.exists():
            raise FileNotFoundError(
                f"Missing feature partition: {partition_dir}"
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

            rows = engineer_file(
                input_path,
                output_path,
            )

            total_files += 1
            total_rows += rows

    print("\n========== ENGINEERING COMPLETE ==========")
    print(f"Files processed: {total_files}")
    print(f"Total rows: {total_rows:,}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
