
from pathlib import Path
from collections import Counter
import pandas as pd
import traceback

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "Dataset" / "set_01"
OUTPUT_DIR = ROOT / "reports" / "set_01" / "original_audit"

CHUNK_SIZE = 200_000

EXPECTED_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]

# ============================================================
# INVENTORY
# ============================================================

def inspect_file(file_path):
    relative_path = file_path.relative_to(DATASET_DIR)

    result = {
        "file": str(relative_path),
        "size_mb": round(file_path.stat().st_size / (1024 ** 2), 3),
        "rows": 0,
        "normal": 0,
        "attack": 0,
        "missing_values": 0,
        "invalid_labels": 0,
        "schema_valid": False,
        "schema": "",
        "status": "OK",
        "error": "",
    }

    try:
        reader = pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            on_bad_lines="error",
        )

        first_chunk = True

        for chunk in reader:
            if first_chunk:
                columns = list(chunk.columns)
                result["schema"] = "|".join(columns)
                result["schema_valid"] = columns == EXPECTED_COLUMNS
                first_chunk = False

            result["rows"] += len(chunk)

            result["missing_values"] += int(
                chunk.isna().sum().sum()
            )

            labels = pd.to_numeric(
                chunk["attack"],
                errors="coerce"
            )

            valid = labels.isin([0, 1])

            result["invalid_labels"] += int((~valid).sum())
            result["normal"] += int((labels == 0).sum())
            result["attack"] += int((labels == 1).sum())

        if first_chunk:
            result["status"] = "EMPTY_FILE"

        elif not result["schema_valid"]:
            result["status"] = "SCHEMA_MISMATCH"

    except Exception as error:
        result["status"] = "READ_ERROR"
        result["error"] = str(error)

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("CAN BUS — ORIGINAL DATASET INVENTORY")
    print("=" * 65)

    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    files = sorted(DATASET_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found.")

    print(f"\nFound {len(files)} CSV files.\n")

    results = []

    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path.name}")

        result = inspect_file(file_path)
        results.append(result)

        print(
            f"    Rows: {result['rows']:,} | "
            f"Normal: {result['normal']:,} | "
            f"Attack: {result['attack']:,} | "
            f"Status: {result['status']}"
        )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = pd.DataFrame(results)

    inventory.to_csv(
        OUTPUT_DIR / "file_inventory.csv",
        index=False
    )

    schema_report = inventory[
        ["file", "schema", "schema_valid", "status", "error"]
    ]

    schema_report.to_csv(
        OUTPUT_DIR / "schema_validation.csv",
        index=False
    )

    partition_report = (
        inventory.groupby(
            inventory["file"].str.split("\\").str[0],
            dropna=False
        )
        .agg(
            files=("file", "count"),
            rows=("rows", "sum"),
            normal=("normal", "sum"),
            attack=("attack", "sum"),
            size_mb=("size_mb", "sum"),
        )
        .reset_index(names="partition")
    )

    partition_report.to_csv(
        OUTPUT_DIR / "partition_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_rows = int(inventory["rows"].sum())
    total_normal = int(inventory["normal"].sum())
    total_attack = int(inventory["attack"].sum())

    errors = inventory[inventory["status"] != "OK"]

    summary = [
        "# Original set_01 dataset inventory",
        "",
        f"- CSV files discovered: {len(files):,}",
        f"- Total records read: {total_rows:,}",
        f"- Normal records: {total_normal:,}",
        f"- Attack records: {total_attack:,}",
        f"- Attack prevalence: "
        f"{100 * total_attack / total_rows:.6f}%"
        if total_rows else "- Attack prevalence: N/A",
        f"- Files with issues: {len(errors):,}",
        "",
        "## Files requiring attention",
        "",
    ]

    if errors.empty:
        summary.append("No file-level read or schema issues detected.")
    else:
        for _, row in errors.iterrows():
            summary.append(
                f"- {row['file']}: {row['status']} — {row['error']}"
            )

    (OUTPUT_DIR / "audit_summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("INVENTORY COMPLETE")
    print("=" * 65)
    print(f"Files:       {len(files):,}")
    print(f"Records:     {total_rows:,}")
    print(f"Normal:      {total_normal:,}")
    print(f"Attack:      {total_attack:,}")
    print(f"Issues:      {len(errors):,}")
    print(f"\nReports saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
