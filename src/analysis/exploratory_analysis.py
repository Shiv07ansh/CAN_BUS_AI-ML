
"""
CAN_BUS_AI — EXPLORATORY DATA ANALYSIS
======================================

Stage 1.3: Class and attack distribution analysis.

READ-ONLY:
- Original CSV files are never modified.
- Reads data in chunks.
- Preserves original file and row ordering.
- Produces CSV reports, figures and a Markdown summary.
"""

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "Dataset" / "set_01"

OUTPUT_DIR = (
    ROOT / "reports" / "set_01" / "stage_1" / "exploratory_analysis"
)

CHUNK_SIZE = 200_000

EXPECTED_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]


# ============================================================
# 2. FILE NAME METADATA
# ============================================================

def extract_file_metadata(file_path):

    relative = file_path.relative_to(DATASET_DIR)

    partition = relative.parts[0]
    filename = file_path.stem

    return {
        "partition": partition,
        "filename": filename,
        "file": str(relative),
    }


# ============================================================
# 3. ANALYSE ONE FILE
# ============================================================

def analyse_file(file_path):

    metadata = extract_file_metadata(file_path)

    normal_count = 0
    attack_count = 0
    invalid_count = 0
    total_count = 0

    block_results = []

    reader = pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
        on_bad_lines="error",
        encoding="utf-8",
    )

    for block_number, chunk in enumerate(reader):

        if list(chunk.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Schema mismatch in {file_path}: "
                f"{list(chunk.columns)}"
            )

        labels = pd.to_numeric(
            chunk["attack"],
            errors="coerce"
        )

        valid = labels.isin([0, 1])

        normal = int((labels[valid] == 0).sum())
        attack = int((labels[valid] == 1).sum())
        invalid = int((~valid).sum())

        block_total = normal + attack

        normal_count += normal
        attack_count += attack
        invalid_count += invalid
        total_count += len(chunk)

        block_results.append({
            **metadata,
            "block_number": block_number + 1,
            "source_row_start": block_number * CHUNK_SIZE,
            "source_row_end": (
                block_number * CHUNK_SIZE + len(chunk) - 1
            ),
            "records": block_total,
            "normal": normal,
            "attack": attack,
            "invalid_labels": invalid,
            "attack_percentage": (
                100 * attack / block_total
                if block_total else 0
            ),
        })

    valid_total = normal_count + attack_count

    file_result = {
        **metadata,
        "records": total_count,
        "normal": normal_count,
        "attack": attack_count,
        "invalid_labels": invalid_count,
        "attack_percentage": (
            100 * attack_count / valid_total
            if valid_total else 0
        ),
        "normal_percentage": (
            100 * normal_count / valid_total
            if valid_total else 0
        ),
        "attack_to_normal_ratio": (
            attack_count / normal_count
            if normal_count else None
        ),
    }

    return file_result, block_results


# ============================================================
# 4. PARTITION SUMMARY
# ============================================================

def build_partition_summary(file_df):

    summary = (
        file_df.groupby("partition")
        .agg(
            files=("file", "count"),
            records=("records", "sum"),
            normal=("normal", "sum"),
            attack=("attack", "sum"),
            invalid_labels=("invalid_labels", "sum"),
        )
        .reset_index()
    )

    summary["attack_percentage"] = (
        100 * summary["attack"] /
        (summary["normal"] + summary["attack"])
    )

    summary["normal_percentage"] = (
        100 * summary["normal"] /
        (summary["normal"] + summary["attack"])
    )

    summary["attack_to_normal_ratio"] = (
        summary["attack"] / summary["normal"]
    )

    return summary


# ============================================================
# 5. FIGURES
# ============================================================

def create_figures(file_df, partition_df, block_df):

    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Figure 1: Class distribution by partition
    # --------------------------------------------------------

    plot_data = partition_df.set_index("partition")[
        ["normal", "attack"]
    ]

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 6),
        logy=True,
    )

    ax.set_title("Normal and Attack Records by Partition")
    ax.set_ylabel("Record count (log scale)")
    ax.set_xlabel("Original partition")
    ax.legend(["Normal", "Attack"])
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(
        figure_dir / "01_partition_class_distribution.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # Figure 2: Attack prevalence by file
    # --------------------------------------------------------

    ordered = file_df.sort_values(
        ["partition", "attack_percentage"]
    )

    plt.figure(figsize=(14, 8))

    plt.barh(
        ordered["file"],
        ordered["attack_percentage"]
    )

    plt.xlabel("Attack records (%)")
    plt.ylabel("Source CSV file")
    plt.title("Attack Prevalence by Original CSV File")
    plt.tight_layout()

    plt.savefig(
        figure_dir / "02_attack_prevalence_by_file.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # Figure 3: Attack prevalence across source-row blocks
    # --------------------------------------------------------

    if not block_df.empty:

        plt.figure(figsize=(14, 6))

        for file_name, group in block_df.groupby("file"):

            plt.plot(
                group["block_number"],
                group["attack_percentage"],
                alpha=0.35,
                linewidth=0.8,
            )

        plt.xlabel("Consecutive source-row block")
        plt.ylabel("Attack prevalence (%)")
        plt.title(
            "Attack Prevalence Across Consecutive "
            "Source-Row Blocks"
        )

        plt.tight_layout()

        plt.savefig(
            figure_dir / "03_attack_prevalence_by_block.png",
            dpi=200
        )
        plt.close()

    # --------------------------------------------------------
    # Figure 4: Attack prevalence distribution
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        file_df["attack_percentage"],
        bins=20,
    )

    plt.xlabel("Attack prevalence per file (%)")
    plt.ylabel("Number of files")
    plt.title("Distribution of Attack Prevalence Across Files")

    plt.tight_layout()

    plt.savefig(
        figure_dir / "04_file_attack_prevalence_histogram.png",
        dpi=200
    )
    plt.close()


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CAN_BUS_AI — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    files = sorted(DATASET_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_results = []
    block_results = []

    for index, file_path in enumerate(files, start=1):

        print(f"\n[{index}/{len(files)}] {file_path.name}")

        file_result, blocks = analyse_file(file_path)

        file_results.append(file_result)
        block_results.extend(blocks)

        print(
            f"    Records: {file_result['records']:,} | "
            f"Attacks: {file_result['attack']:,} | "
            f"Attack rate: "
            f"{file_result['attack_percentage']:.4f}%"
        )

    file_df = pd.DataFrame(file_results)
    block_df = pd.DataFrame(block_results)

    partition_df = build_partition_summary(file_df)

    # --------------------------------------------------------
    # SAVE TABLES
    # --------------------------------------------------------

    file_df.to_csv(
        OUTPUT_DIR / "01_file_attack_distribution.csv",
        index=False
    )

    partition_df.to_csv(
        OUTPUT_DIR / "02_partition_attack_distribution.csv",
        index=False
    )

    block_df.to_csv(
        OUTPUT_DIR / "03_source_block_attack_distribution.csv",
        index=False
    )

    # --------------------------------------------------------
    # SAVE FIGURES
    # --------------------------------------------------------

    create_figures(
        file_df,
        partition_df,
        block_df
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_normal = int(file_df["normal"].sum())
    total_attack = int(file_df["attack"].sum())
    total = total_normal + total_attack

    attack_free_files = int(
        (file_df["attack"] == 0).sum()
    )

    nonzero_attack_files = int(
        (file_df["attack"] > 0).sum()
    )

    summary = [
        "# Stage 1.3 — Class and Attack Distribution",
        "",
        "## Overall distribution",
        "",
        f"- Files analysed: {len(files):,}",
        f"- Total valid records: {total:,}",
        f"- Normal records: {total_normal:,}",
        f"- Attack records: {total_attack:,}",
        f"- Attack prevalence: "
        f"{100 * total_attack / total:.6f}%",
        "",
        "## File representation",
        "",
        f"- Files containing attacks: {nonzero_attack_files}",
        f"- Files containing no attacks: {attack_free_files}",
        f"- Files with highest attack prevalence: "
        f"{file_df.loc[file_df['attack_percentage'].idxmax(), 'file']}",
        "",
        "## Partition distribution",
        "",
        partition_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The attack column provides binary labels only. "
        "Attack types cannot be distinguished from these labels alone.",
        "",
        "Attack prevalence across source-row blocks is a descriptive "
        "diagnostic. It does not by itself establish attack episode "
        "boundaries or attack-generation methodology.",
        "",
        "Original CSV files were not modified.",
    ]

    (OUTPUT_DIR / "analysis_summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("EXPLORATORY ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"Normal records: {total_normal:,}")
    print(f"Attack records: {total_attack:,}")
    print(f"Attack prevalence: {100 * total_attack / total:.6f}%")
    print(f"Files containing attacks: {nonzero_attack_files}")
    print(f"Files without attacks: {attack_free_files}")

    print(f"\nReports saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
