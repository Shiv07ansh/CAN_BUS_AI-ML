
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "Dataset" / "set_01"

OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "sampled_stratified" / "set_01"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_2"
    / "stratified_sampling"
)

SAMPLE_FRACTION = 0.30
RANDOM_SEED = 42
CHUNK_SIZE = 200_000

LABEL_COLUMN = "attack"


# ============================================================
# REPRODUCIBLE PER-FILE RANDOM GENERATOR
# ============================================================

def make_file_rng(relative_path):
    """
    Create a deterministic RNG for each file.
    File selection does not depend on processing order.
    """
    seed_text = f"{RANDOM_SEED}:{relative_path.as_posix()}"

    digest = hashlib.sha256(
        seed_text.encode("utf-8")
    ).digest()

    file_seed = int.from_bytes(digest[:8], "little")

    return np.random.default_rng(file_seed)


# ============================================================
# PASS 1: FIND SOURCE ROW POSITIONS BY CLASS
# ============================================================

def collect_class_positions(file_path):
    """
    Collect original zero-based row positions separately
    for normal and attack records.
    """

    positions = {
        0: [],
        1: [],
    }

    row_offset = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=[LABEL_COLUMN],
        dtype={LABEL_COLUMN: "int8"},
        chunksize=CHUNK_SIZE,
    ):

        labels = chunk[LABEL_COLUMN].to_numpy(
            dtype=np.int8
        )

        chunk_positions = np.arange(
            row_offset,
            row_offset + len(labels),
            dtype=np.int64,
        )

        for class_label in (0, 1):
            class_positions = chunk_positions[
                labels == class_label
            ]

            if len(class_positions):
                positions[class_label].append(
                    class_positions
                )

        row_offset += len(labels)

    for class_label in (0, 1):
        if positions[class_label]:
            positions[class_label] = np.concatenate(
                positions[class_label]
            )
        else:
            positions[class_label] = np.array(
                [],
                dtype=np.int64,
            )

    return positions, row_offset


# ============================================================
# SELECT EXACT NUMBER OF ROWS PER CLASS
# ============================================================

def select_class_rows(positions, rng):
    selected = {}
    details = {}

    for class_label in (0, 1):
        available = positions[class_label]

        if len(available) == 0:
            selected[class_label] = np.array(
                [],
                dtype=np.int64,
            )

            details[class_label] = {
                "original_records": 0,
                "selected_records": 0,
            }
            continue

        target_count = max(
            1,
            int(round(len(available) * SAMPLE_FRACTION)),
        )

        target_count = min(
            target_count,
            len(available),
        )

        chosen = rng.choice(
            available,
            size=target_count,
            replace=False,
        )

        selected[class_label] = np.sort(chosen)

        details[class_label] = {
            "original_records": len(available),
            "selected_records": len(chosen),
        }

    return selected, details


# ============================================================
# PASS 2: WRITE SELECTED ROWS IN ORIGINAL ORDER
# ============================================================

def write_sampled_file(
    source_path,
    destination_path,
    selected_positions,
):
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination_path.exists():
        destination_path.unlink()

    selected_all = np.sort(
        np.concatenate([
            selected_positions[0],
            selected_positions[1],
        ])
    )

    row_offset = 0
    output_records = 0
    first_write = True

    for chunk in pd.read_csv(
        source_path,
        dtype="string",
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        chunk_end = row_offset + len(chunk)

        left = np.searchsorted(
            selected_all,
            row_offset,
            side="left",
        )

        right = np.searchsorted(
            selected_all,
            chunk_end,
            side="left",
        )

        selected_in_chunk = selected_all[left:right]

        if len(selected_in_chunk):
            local_positions = (
                selected_in_chunk - row_offset
            )

            sampled_chunk = chunk.iloc[
                local_positions
            ]

            sampled_chunk.to_csv(
                destination_path,
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )

            output_records += len(sampled_chunk)
            first_write = False

        row_offset = chunk_end

    return output_records


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path):
    relative_path = file_path.relative_to(DATASET_ROOT)

    partition = relative_path.parent.as_posix()
    file_name = relative_path.name

    print(f"\nProcessing: {relative_path}")

    # Pass 1: collect source row positions.
    positions, original_records = collect_class_positions(
        file_path
    )

    rng = make_file_rng(relative_path)

    selected_positions, details = select_class_rows(
        positions,
        rng,
    )

    destination_path = OUTPUT_ROOT / relative_path

    # Pass 2: write selected rows.
    sampled_records = write_sampled_file(
        file_path,
        destination_path,
        selected_positions,
    )

    original_normal = details[0]["original_records"]
    original_attack = details[1]["original_records"]

    sampled_normal = details[0]["selected_records"]
    sampled_attack = details[1]["selected_records"]

    expected_records = sampled_normal + sampled_attack

    if sampled_records != expected_records:
        raise RuntimeError(
            f"Output count mismatch for {file_name}: "
            f"expected {expected_records}, "
            f"wrote {sampled_records}"
        )

    original_attack_rate = (
        100 * original_attack / original_records
        if original_records else 0
    )

    sampled_attack_rate = (
        100 * sampled_attack / sampled_records
        if sampled_records else 0
    )

    summary = {
        "partition": partition,
        "file": file_name,
        "original_records": original_records,
        "sampled_records": sampled_records,
        "retained_percentage": (
            100 * sampled_records / original_records
            if original_records else 0
        ),
        "original_normal": original_normal,
        "sampled_normal": sampled_normal,
        "original_attack": original_attack,
        "sampled_attack": sampled_attack,
        "original_attack_percentage": original_attack_rate,
        "sampled_attack_percentage": sampled_attack_rate,
        "normal_retention_percentage": (
            100 * sampled_normal / original_normal
            if original_normal else 0
        ),
        "attack_retention_percentage": (
            100 * sampled_attack / original_attack
            if original_attack else 0
        ),
        "output_file": str(
            destination_path.relative_to(PROJECT_ROOT)
        ),
    }

    print(
        f"  Original: {original_records:,}\n"
        f"  Sampled:  {sampled_records:,}\n"
        f"  Retained: {summary['retained_percentage']:.2f}%\n"
        f"  Original attack rate: "
        f"{original_attack_rate:.5f}%\n"
        f"  Sampled attack rate:  "
        f"{sampled_attack_rate:.5f}%"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    files = sorted(DATASET_ROOT.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files found."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    print("=" * 65)
    print("STRATIFIED ROW SAMPLING — SET 01")
    print("=" * 65)
    print(f"Files: {len(files)}")
    print(f"Sampling fraction: {SAMPLE_FRACTION:.0%}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    for file_path in files:
        summaries.append(
            process_file(file_path)
        )

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        REPORT_ROOT / "01_file_sampling_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Partition-level summary
    # --------------------------------------------------------

    partition_df = (
        summary_df.groupby("partition")
        .agg(
            files=("file", "count"),
            original_records=("original_records", "sum"),
            sampled_records=("sampled_records", "sum"),
            original_normal=("original_normal", "sum"),
            sampled_normal=("sampled_normal", "sum"),
            original_attack=("original_attack", "sum"),
            sampled_attack=("sampled_attack", "sum"),
        )
        .reset_index()
    )

    partition_df["retained_percentage"] = (
        100 * partition_df["sampled_records"]
        / partition_df["original_records"]
    )

    partition_df["original_attack_percentage"] = (
        100 * partition_df["original_attack"]
        / partition_df["original_records"]
    )

    partition_df["sampled_attack_percentage"] = (
        100 * partition_df["sampled_attack"]
        / partition_df["sampled_records"]
    )

    partition_df.to_csv(
        REPORT_ROOT / "02_partition_sampling_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Configuration record
    # --------------------------------------------------------

    config = {
        "dataset": "can-train-and-test Set 01",
        "sampling_method": (
            "Independent simple random sampling without "
            "replacement within each class and original CSV file."
        ),
        "sample_fraction": SAMPLE_FRACTION,
        "random_seed": RANDOM_SEED,
        "chunk_size": CHUNK_SIZE,
        "class_column": LABEL_COLUMN,
        "source_row_order": "Preserved",
        "original_files_modified": False,
        "output_directory": str(
            OUTPUT_ROOT.relative_to(PROJECT_ROOT)
        ),
    }

    with open(
        REPORT_ROOT / "sampling_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=4)

    print("\n" + "=" * 65)
    print("SAMPLING COMPLETE")
    print("=" * 65)

    print("\nPartition summary:")
    print(partition_df.to_string(index=False))

    print("\nSampled data:")
    print(OUTPUT_ROOT)

    print("\nReports:")
    print(REPORT_ROOT)


if __name__ == "__main__":
    main()
