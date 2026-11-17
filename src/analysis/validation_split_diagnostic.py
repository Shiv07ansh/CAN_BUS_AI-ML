
from pathlib import Path
import json
import pandas as pd
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_ROOT = (
    ROOT / "data" / "sampled_stratified"
    / "set_01" / "train_01"
)

OUTPUT_ROOT = (
    ROOT / "reports" / "set_01"
    / "stage_4" / "split_diagnostic"
)

CHUNK_SIZE = 200_000

TRAIN_FRACTION = 0.80
TEMPORAL_BINS = 5

# =========================================================
# ANALYSE ONE FILE
# =========================================================

def analyse_file(path):

    df = pd.read_csv(
        path,
        dtype={
            "timestamp": "float64",
            "attack": "int8",
        },
        usecols=["timestamp", "attack"],
    )

    n = len(df)

    if n == 0:
        return None, []

    labels = df["attack"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    attack_total = int(labels.sum())
    normal_total = n - attack_total

    split_index = int(n * TRAIN_FRACTION)

    # -----------------------------------------------------
    # File-level statistics
    # -----------------------------------------------------

    file_result = {
        "file": path.name,
        "records": n,
        "normal": normal_total,
        "attack": attack_total,
        "attack_percentage": 100 * attack_total / n,
        "contains_attacks": attack_total > 0,
        "first_timestamp": float(timestamps[0]),
        "last_timestamp": float(timestamps[-1]),
        "timestamp_decreases": int(
            np.sum(np.diff(timestamps) < 0)
        ),
    }

    # -----------------------------------------------------
    # Chronological 80/20 split
    # -----------------------------------------------------

    train_labels = labels[:split_index]
    validation_labels = labels[split_index:]

    train_attack = int(train_labels.sum())
    validation_attack = int(validation_labels.sum())

    file_result.update({
        "chronological_train_records": len(train_labels),
        "chronological_train_attacks": train_attack,
        "chronological_train_attack_percentage": (
            100 * train_attack / len(train_labels)
            if len(train_labels) else 0
        ),
        "chronological_validation_records": len(validation_labels),
        "chronological_validation_attacks": validation_attack,
        "chronological_validation_attack_percentage": (
            100 * validation_attack / len(validation_labels)
            if len(validation_labels) else 0
        ),
    })

    # -----------------------------------------------------
    # Five consecutive temporal sections
    # -----------------------------------------------------

    bin_edges = np.linspace(
        0, n, TEMPORAL_BINS + 1, dtype=int
    )

    temporal_results = []

    for i in range(TEMPORAL_BINS):

        start = bin_edges[i]
        end = bin_edges[i + 1]

        segment_labels = labels[start:end]
        segment_timestamps = timestamps[start:end]

        segment_count = len(segment_labels)
        segment_attacks = int(segment_labels.sum())

        temporal_results.append({
            "file": path.name,
            "temporal_bin": i + 1,
            "start_row": start,
            "end_row_exclusive": end,
            "records": segment_count,
            "normal": segment_count - segment_attacks,
            "attack": segment_attacks,
            "attack_percentage": (
                100 * segment_attacks / segment_count
                if segment_count else 0
            ),
            "first_timestamp": (
                float(segment_timestamps[0])
                if segment_count else None
            ),
            "last_timestamp": (
                float(segment_timestamps[-1])
                if segment_count else None
            ),
        })

    return file_result, temporal_results


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(TRAIN_ROOT.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No training CSV files found: {TRAIN_ROOT}"
        )

    file_results = []
    temporal_results = []

    print("=" * 65)
    print("TRAINING VALIDATION SPLIT DIAGNOSTIC")
    print("=" * 65)

    for index, path in enumerate(files, start=1):

        print(f"\n[{index}/{len(files)}] {path.name}")

        file_result, temporal = analyse_file(path)

        if file_result is None:
            continue

        file_results.append(file_result)
        temporal_results.extend(temporal)

        print(
            f"  Rows: {file_result['records']:,} | "
            f"Attacks: {file_result['attack']:,} | "
            f"Attack rate: "
            f"{file_result['attack_percentage']:.4f}%"
        )

        print(
            f"  Chronological validation attacks: "
            f"{file_result['chronological_validation_attacks']:,}"
        )

    # -----------------------------------------------------
    # Save file-level report
    # -----------------------------------------------------

    file_df = pd.DataFrame(file_results)

    file_df.to_csv(
        OUTPUT_ROOT / "01_file_level_diagnostic.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Save temporal report
    # -----------------------------------------------------

    temporal_df = pd.DataFrame(temporal_results)

    temporal_df.to_csv(
        OUTPUT_ROOT / "02_temporal_distribution.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Aggregate chronological split
    # -----------------------------------------------------

    total_train = int(
        file_df["chronological_train_records"].sum()
    )

    total_train_attack = int(
        file_df["chronological_train_attacks"].sum()
    )

    total_validation = int(
        file_df["chronological_validation_records"].sum()
    )

    total_validation_attack = int(
        file_df["chronological_validation_attacks"].sum()
    )

    total_records = total_train + total_validation

    summary = {
        "training_files": len(file_df),
        "total_records": total_records,
        "total_attacks": int(file_df["attack"].sum()),
        "attack_containing_files": int(
            file_df["contains_attacks"].sum()
        ),
        "zero_attack_files": int(
            (~file_df["contains_attacks"]).sum()
        ),
        "files_with_zero_attacks_in_chronological_validation": int(
            (file_df["chronological_validation_attacks"] == 0).sum()
        ),
        "files_with_zero_attacks_in_chronological_training": int(
            (file_df["chronological_train_attacks"] == 0).sum()
        ),
        "chronological_train": {
            "records": total_train,
            "attacks": total_train_attack,
            "attack_percentage": (
                100 * total_train_attack / total_train
            ),
        },
        "chronological_validation": {
            "records": total_validation,
            "attacks": total_validation_attack,
            "attack_percentage": (
                100 * total_validation_attack / total_validation
            ),
        },
        "temporal_bins_per_file": TEMPORAL_BINS,
        "chronological_split_fraction": TRAIN_FRACTION,
        "note": (
            "Chronological split is applied independently within "
            "each file, preserving file order. Timestamp decreases "
            "are reported rather than automatically corrected."
        ),
    }

    with open(
        OUTPUT_ROOT / "03_split_diagnostic_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 65)

    print(f"Training files: {len(file_df)}")
    print(f"Total records: {total_records:,}")
    print(
        f"Attack-containing files: "
        f"{summary['attack_containing_files']}"
    )

    print("\nChronological 80/20 split:")
    print(
        f"Training:   {total_train:,} records | "
        f"{total_train_attack:,} attacks"
    )
    print(
        f"Validation: {total_validation:,} records | "
        f"{total_validation_attack:,} attacks"
    )

    print(
        "\nFiles with zero validation attacks: "
        f"{summary['files_with_zero_attacks_in_chronological_validation']}"
    )

    print(f"\nReports saved to:\n{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
