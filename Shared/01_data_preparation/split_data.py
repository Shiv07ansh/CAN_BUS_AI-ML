
from pathlib import Path
import json

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT.parent / "data" / "sampled_stratified" / "set_01"
TRAIN_DIR = DATASET_DIR / "train_01"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "set_01"
    / "split"
)

TRAIN_FILES = [
    "accessory-2.csv",
    "attack-free-1.csv",
    "attack-free-2.csv",
    "DoS-2.csv",
    "force-neutral-2.csv",
    "rpm-1.csv",
    "standstill-1.csv",
]

VALIDATION_FILES = [
    "accessory-1.csv",
    "DoS-1.csv",
    "force-neutral-1.csv",
    "rpm-2.csv",
    "standstill-2.csv",
]


# ============================================================
# MAIN
# ============================================================

def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training partition not found: {TRAIN_DIR}"
        )

    actual_files = {
        path.name for path in TRAIN_DIR.glob("*.csv")
    }

    expected_files = set(TRAIN_FILES + VALIDATION_FILES)

    missing_files = expected_files - actual_files
    unexpected_files = actual_files - expected_files

    if missing_files or unexpected_files:
        raise ValueError(
            "Recording inventory does not match "
            "the established experiment.\n"
            f"Missing files: {sorted(missing_files)}\n"
            f"Unexpected files: {sorted(unexpected_files)}"
        )

    if set(TRAIN_FILES) & set(VALIDATION_FILES):
        raise ValueError("Training/validation overlap detected.")

    assignments = []

    for filename in TRAIN_FILES:
        assignments.append({
            "file": filename,
            "split": "train",
        })

    for filename in VALIDATION_FILES:
        assignments.append({
            "file": filename,
            "split": "validation",
        })

    assignment_df = pd.DataFrame(assignments)

    training_df = assignment_df[
        assignment_df["split"] == "train"
    ][["file"]]

    validation_df = assignment_df[
        assignment_df["split"] == "validation"
    ][["file"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assignment_df.to_csv(
        OUTPUT_DIR / "split_assignment.csv",
        index=False,
    )

    training_df.to_csv(
        OUTPUT_DIR / "training_files.csv",
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation_files.csv",
        index=False,
    )

    summary = {
        "split_method": "Frozen recording-level split",
        "random_state": 42,
        "training_recordings": len(TRAIN_FILES),
        "validation_recordings": len(VALIDATION_FILES),
        "total_recordings": len(actual_files),
        "overlap": False,
        "training_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "official_test_partitions_used": False,
    }

    with open(
        OUTPUT_DIR / "split_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== FROZEN SPLIT ==========")
    print(f"Training recordings: {len(TRAIN_FILES)}")
    print(f"Validation recordings: {len(VALIDATION_FILES)}")
    print("Recording overlap: None")
    print("\nSplit files saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
