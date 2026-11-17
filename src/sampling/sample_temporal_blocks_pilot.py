
from pathlib import Path
import math
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "Dataset" / "set_01"

SAMPLED_ROOT = (
    PROJECT_ROOT / "data" / "sampled" / "set_01"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_2"
    / "sampling"
)

BLOCK_SIZE = 10_000
SAMPLE_FRACTION = 0.30
RANDOM_SEED = 42
CHUNK_SIZE = 200_000

EXPECTED_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]


# ============================================================
# PASS 1: BUILD BLOCK INVENTORY
# ============================================================

def build_block_inventory(file_path):
    """
    Count records and attack records in each contiguous
    source-row block without loading the entire CSV.
    """

    block_info = {}
    row_offset = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=["attack"],
        dtype={"attack": "int8"},
        chunksize=CHUNK_SIZE,
    ):

        labels = chunk["attack"].to_numpy(dtype=np.int8)

        row_positions = np.arange(
            row_offset,
            row_offset + len(labels),
            dtype=np.int64,
        )

        block_ids = row_positions // BLOCK_SIZE

        frame = pd.DataFrame({
            "block_id": block_ids,
            "attack": labels,
        })

        grouped = frame.groupby("block_id")["attack"].agg(
            records="size",
            attacks="sum",
        )

        for block_id, stats in grouped.iterrows():
            block_id = int(block_id)

            if block_id not in block_info:
                block_info[block_id] = {
                    "records": 0,
                    "attacks": 0,
                }

            block_info[block_id]["records"] += int(
                stats["records"]
            )
            block_info[block_id]["attacks"] += int(
                stats["attacks"]
            )

        row_offset += len(chunk)

    return block_info, row_offset


# ============================================================
# SELECT BLOCKS
# ============================================================

def select_blocks(block_info, rng):
    """
    Select approximately 30% of blocks independently from
    the attack-containing and zero-attack strata.

    At least one block is selected from each non-empty stratum.
    """

    attack_blocks = [
        block_id
        for block_id, stats in block_info.items()
        if stats["attacks"] > 0
    ]

    zero_attack_blocks = [
        block_id
        for block_id, stats in block_info.items()
        if stats["attacks"] == 0
    ]

    selected = set()

    strata = {
        "attack_containing": attack_blocks,
        "zero_attack": zero_attack_blocks,
    }

    selection_details = {}

    for stratum_name, block_ids in strata.items():

        if not block_ids:
            selection_details[stratum_name] = {
                "available_blocks": 0,
                "selected_blocks": 0,
            }
            continue

        number_to_select = max(
            1,
            math.ceil(len(block_ids) * SAMPLE_FRACTION),
        )

        chosen = rng.choice(
            np.asarray(block_ids, dtype=np.int64),
            size=number_to_select,
            replace=False,
        )

        chosen = set(int(value) for value in chosen)

        selected.update(chosen)

        selection_details[stratum_name] = {
            "available_blocks": len(block_ids),
            "selected_blocks": len(chosen),
        }

    return selected, selection_details


# ============================================================
# PASS 2: WRITE SELECTED BLOCKS
# ============================================================

def write_sampled_file(
    source_path,
    destination_path,
    selected_blocks,
):
    """
    Copy selected complete blocks into a new CSV.

    All columns are read as strings to avoid unnecessary
    type conversion or modification of source representations.
    """

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination_path.exists():
        destination_path.unlink()

    row_offset = 0
    output_records = 0
    first_write = True

    for chunk in pd.read_csv(
        source_path,
        dtype="string",
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        row_positions = np.arange(
            row_offset,
            row_offset + len(chunk),
            dtype=np.int64,
        )

        block_ids = row_positions // BLOCK_SIZE

        mask = np.isin(
            block_ids,
            list(selected_blocks),
        )

        selected_chunk = chunk.loc[mask]

        if not selected_chunk.empty:
            selected_chunk.to_csv(
                destination_path,
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )

            output_records += len(selected_chunk)
            first_write = False

        row_offset += len(chunk)

    return output_records


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path, rng):
    partition = file_path.parent.name
    file_name = file_path.name

    print(f"\nProcessing {partition}/{file_name}")

    block_info, original_records = build_block_inventory(
        file_path
    )

    selected_blocks, selection_details = select_blocks(
        block_info,
        rng,
    )

    destination_path = (
        SAMPLED_ROOT / partition / file_name
    )

    sampled_records = write_sampled_file(
        file_path,
        destination_path,
        selected_blocks,
    )

    original_attacks = sum(
        stats["attacks"]
        for stats in block_info.values()
    )

    original_normal = original_records - original_attacks

    selected_attacks = sum(
        block_info[block_id]["attacks"]
        for block_id in selected_blocks
    )

    selected_records_expected = sum(
        block_info[block_id]["records"]
        for block_id in selected_blocks
    )

    selected_normal = (
        selected_records_expected - selected_attacks
    )

    # --------------------------------------------------------
    # Per-block selection manifest
    # --------------------------------------------------------

    block_manifest = []

    for block_id, stats in sorted(block_info.items()):
        records = stats["records"]
        attacks = stats["attacks"]

        block_manifest.append({
            "partition": partition,
            "file": file_name,
            "block_id": block_id,
            "source_start_row": block_id * BLOCK_SIZE,
            "source_end_row_exclusive": (
                block_id * BLOCK_SIZE + records
            ),
            "records": records,
            "normal_records": records - attacks,
            "attack_records": attacks,
            "attack_percentage": (
                100.0 * attacks / records
                if records else 0.0
            ),
            "stratum": (
                "attack_containing"
                if attacks > 0
                else "zero_attack"
            ),
            "selected": block_id in selected_blocks,
        })

    # --------------------------------------------------------
    # Per-file summary
    # --------------------------------------------------------

    summary = {
        "partition": partition,
        "file": file_name,
        "original_records": original_records,
        "sampled_records": sampled_records,
        "sample_fraction_percentage": (
            100.0 * sampled_records / original_records
            if original_records else 0.0
        ),
        "original_normal": original_normal,
        "sampled_normal": selected_normal,
        "original_attack": original_attacks,
        "sampled_attack": selected_attacks,
        "original_attack_percentage": (
            100.0 * original_attacks / original_records
            if original_records else 0.0
        ),
        "sampled_attack_percentage": (
            100.0 * selected_attacks / sampled_records
            if sampled_records else 0.0
        ),
        "total_blocks": len(block_info),
        "selected_blocks": len(selected_blocks),
        "attack_blocks_available": (
            selection_details["attack_containing"][
                "available_blocks"
            ]
        ),
        "attack_blocks_selected": (
            selection_details["attack_containing"][
                "selected_blocks"
            ]
        ),
        "zero_attack_blocks_available": (
            selection_details["zero_attack"][
                "available_blocks"
            ]
        ),
        "zero_attack_blocks_selected": (
            selection_details["zero_attack"][
                "selected_blocks"
            ]
        ),
        "output_file": str(destination_path.relative_to(
            PROJECT_ROOT
        )),
    }

    print(
        f"  Original records: {original_records:,}\n"
        f"  Sampled records:  {sampled_records:,}\n"
        f"  Retained:         "
        f"{summary['sample_fraction_percentage']:.2f}%\n"
        f"  Original attack:  "
        f"{summary['original_attack_percentage']:.4f}%\n"
        f"  Sampled attack:   "
        f"{summary['sampled_attack_percentage']:.4f}%"
    )

    if sampled_records != selected_records_expected:
        raise RuntimeError(
            f"Record-count mismatch in {file_name}: "
            f"expected {selected_records_expected}, "
            f"wrote {sampled_records}"
        )

    return summary, block_manifest


# ============================================================
# MAIN
# ============================================================

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {DATASET_ROOT}"
        )

    csv_files = sorted(DATASET_ROOT.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in the original dataset."
        )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Fixed seed and deterministic sorted file order.
    rng = np.random.default_rng(RANDOM_SEED)

    file_summaries = []
    all_block_manifests = []

    print("=" * 70)
    print("REPRODUCIBLE TEMPORAL-BLOCK SAMPLING — SET 01")
    print("=" * 70)
    print(f"Files: {len(csv_files)}")
    print(f"Block size: {BLOCK_SIZE:,}")
    print(f"Sampling fraction: {SAMPLE_FRACTION:.0%}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    for file_path in csv_files:
        summary, block_manifest = process_file(
            file_path,
            rng,
        )

        file_summaries.append(summary)
        all_block_manifests.extend(block_manifest)

    # --------------------------------------------------------
    # Write manifests
    # --------------------------------------------------------

    summary_df = pd.DataFrame(file_summaries)

    summary_df.to_csv(
        REPORT_ROOT / "01_sampling_file_summary.csv",
        index=False,
    )

    block_df = pd.DataFrame(all_block_manifests)

    block_df.to_csv(
        REPORT_ROOT / "02_sampling_block_manifest.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Partition-level summary
    # --------------------------------------------------------

    partition_summary = (
        summary_df.groupby("partition")
        .agg(
            original_records=("original_records", "sum"),
            sampled_records=("sampled_records", "sum"),
            original_normal=("original_normal", "sum"),
            sampled_normal=("sampled_normal", "sum"),
            original_attack=("original_attack", "sum"),
            sampled_attack=("sampled_attack", "sum"),
            files=("file", "count"),
        )
        .reset_index()
    )

    partition_summary["sample_fraction_percentage"] = (
        100.0
        * partition_summary["sampled_records"]
        / partition_summary["original_records"]
    )

    partition_summary["original_attack_percentage"] = (
        100.0
        * partition_summary["original_attack"]
        / partition_summary["original_records"]
    )

    partition_summary["sampled_attack_percentage"] = (
        100.0
        * partition_summary["sampled_attack"]
        / partition_summary["sampled_records"]
    )

    partition_summary.to_csv(
        REPORT_ROOT / "03_sampling_partition_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Configuration record
    # --------------------------------------------------------

    config = {
        "dataset": "can-train-and-test Set 01",
        "block_size": BLOCK_SIZE,
        "sample_fraction": SAMPLE_FRACTION,
        "random_seed": RANDOM_SEED,
        "chunk_size": CHUNK_SIZE,
        "selection_method": (
            "Independent random selection of approximately "
            "30 percent of attack-containing and zero-attack "
            "temporal blocks within each original CSV file. "
            "At least one block is selected from each "
            "non-empty stratum."
        ),
        "row_order": "Preserved",
        "original_files_modified": False,
    }

    with open(
        REPORT_ROOT / "sampling_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=4)

    print("\n" + "=" * 70)
    print("SAMPLING COMPLETED")
    print("=" * 70)
    print("\nPartition summary:")
    print(partition_summary.to_string(index=False))

    print("\nSampled data directory:")
    print(SAMPLED_ROOT)

    print("\nSampling reports:")
    print(REPORT_ROOT)


if __name__ == "__main__":
    main()
