
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "set_01"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_2"
    / "block_size_assessment"
)

CHUNK_SIZE = 200_000

BLOCK_SIZES = [5_000, 10_000, 20_000]


# ============================================================
# PROCESS ONE FILE
# ============================================================

def assess_file(file_path):
    partition = file_path.parent.name
    file_name = file_path.name

    print(f"Assessing: {partition}/{file_name}")

    row_offset = 0
    block_results = []

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

        for block_size in BLOCK_SIZES:
            block_ids = row_positions // block_size

            frame = pd.DataFrame({
                "block_id": block_ids,
                "attack": labels,
            })

            grouped = frame.groupby("block_id")["attack"].agg(
                records="size",
                attack_records="sum",
            )

            for block_id, stats in grouped.iterrows():
                records = int(stats["records"])
                attacks = int(stats["attack_records"])

                block_start = int(block_id) * block_size
                block_end = block_start + records

                block_results.append({
                    "partition": partition,
                    "file": file_name,
                    "block_size": block_size,
                    "block_id": int(block_id),
                    "source_start_row": block_start,
                    "source_end_row_exclusive": block_end,
                    "records": records,
                    "normal_records": records - attacks,
                    "attack_records": attacks,
                    "attack_percentage": (
                        100.0 * attacks / records
                        if records else 0.0
                    ),
                    "contains_attack": attacks > 0,
                })

        row_offset += len(chunk)

    return block_results


# ============================================================
# SUMMARISE CANDIDATE BLOCK SIZES
# ============================================================

def summarise_blocks(block_df):
    summary_rows = []

    group_columns = [
        "partition",
        "file",
        "block_size",
    ]

    for (partition, file_name, block_size), group in (
        block_df.groupby(group_columns)
    ):

        total_blocks = len(group)
        total_records = int(group["records"].sum())
        total_attacks = int(group["attack_records"].sum())

        positive_blocks = group[
            group["contains_attack"]
        ]

        # Hypothetical upper-bound coverage:
        # select the most attack-dense 30% of blocks.
        # This is diagnostic only, NOT our sampling strategy.
        number_to_select = max(
            1,
            int(np.ceil(total_blocks * 0.30)),
        )

        top_blocks = group.nlargest(
            number_to_select,
            "attack_records",
        )

        top_block_attack_coverage = (
            100.0 * top_blocks["attack_records"].sum()
            / total_attacks
            if total_attacks else 0.0
        )

        summary_rows.append({
            "partition": partition,
            "file": file_name,
            "block_size": block_size,
            "total_blocks": total_blocks,
            "total_records": total_records,
            "total_attack_records": total_attacks,
            "attack_prevalence_percentage": (
                100.0 * total_attacks / total_records
                if total_records else 0.0
            ),
            "blocks_containing_attacks": len(positive_blocks),
            "blocks_without_attacks": (
                total_blocks - len(positive_blocks)
            ),
            "percentage_blocks_containing_attacks": (
                100.0 * len(positive_blocks) / total_blocks
                if total_blocks else 0.0
            ),
            "median_attack_percentage_positive_blocks": (
                positive_blocks["attack_percentage"].median()
                if len(positive_blocks) else 0.0
            ),
            "maximum_block_attack_percentage": (
                group["attack_percentage"].max()
                if total_blocks else 0.0
            ),
            "attack_coverage_top_30pct_blocks": (
                top_block_attack_coverage
            ),
            "final_block_records": int(
                group.iloc[-1]["records"]
            ),
        })

    return pd.DataFrame(summary_rows)


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
            f"No CSV files found under: {DATASET_ROOT}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_blocks = []

    print("=" * 65)
    print("TEMPORAL BLOCK-SIZE ASSESSMENT — SET 01")
    print("=" * 65)
    print(f"Files: {len(files)}")
    print(f"Block sizes: {BLOCK_SIZES}")
    print()

    for file_path in files:
        all_blocks.extend(assess_file(file_path))

    block_df = pd.DataFrame(all_blocks)

    block_df.to_csv(
        OUTPUT_DIR / "01_block_level_distribution.csv",
        index=False,
    )

    summary_df = summarise_blocks(block_df)

    summary_df.to_csv(
        OUTPUT_DIR / "02_block_size_comparison.csv",
        index=False,
    )

    # Overall comparison by block size
    overall_rows = []

    for block_size, group in block_df.groupby("block_size"):
        total_blocks = len(group)
        positive_blocks = int(group["contains_attack"].sum())
        total_attacks = int(group["attack_records"].sum())

        number_to_select = max(
            1,
            int(np.ceil(total_blocks * 0.30)),
        )

        top_blocks = group.nlargest(
            number_to_select,
            "attack_records",
        )

        overall_rows.append({
            "block_size": block_size,
            "total_blocks": total_blocks,
            "blocks_containing_attacks": positive_blocks,
            "blocks_without_attacks": (
                total_blocks - positive_blocks
            ),
            "percentage_blocks_containing_attacks": (
                100.0 * positive_blocks / total_blocks
                if total_blocks else 0.0
            ),
            "total_attack_records": total_attacks,
            "attack_coverage_top_30pct_blocks": (
                100.0 * top_blocks["attack_records"].sum()
                / total_attacks
                if total_attacks else 0.0
            ),
        })

    pd.DataFrame(overall_rows).to_csv(
        OUTPUT_DIR / "03_overall_block_size_comparison.csv",
        index=False,
    )

    print("\nAssessment complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nOverall comparison:")
    print(pd.DataFrame(overall_rows).to_string(index=False))


if __name__ == "__main__":
    main()
