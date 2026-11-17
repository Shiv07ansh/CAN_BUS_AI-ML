
"""
CAN_BUS_AI — ORIGINAL DATA QUALITY AUDIT
========================================

Stage 1.2: Inspect original set_01 CSV files.

READ-ONLY:
- Never modifies source CSV files.
- Reads in chunks to control memory usage.
- Writes audit reports only.

Checks:
1. CSV readability and schema
2. Missing / blank fields
3. Invalid labels
4. Timestamp validity and ordering
5. Arbitration ID validity
6. Payload validity and length
7. Exact duplicates within chunks and adjacent chunk boundaries
8. Conflicting labels within chunks
"""

from pathlib import Path
from collections import Counter
import re

import numpy as np
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "Dataset" / "set_01"
OUTPUT_DIR = (
    ROOT / "reports" / "set_01" / "stage_1" / "data_quality"
)

CHUNK_SIZE = 200_000

EXPECTED_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]

# Classical CAN supports 11-bit IDs.
# Extended CAN supports 29-bit IDs.
MAX_CAN_ID = 0x1FFFFFFF

HEX_PATTERN = re.compile(r"^(?:0x)?[0-9a-fA-F]+$")

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def is_blank(value):
    return str(value).strip() == ""

def parse_can_id(value):
    """
    Return (valid, numeric_id, error_reason).
    Accepts hexadecimal IDs with or without 0x prefix.
    """

    value = str(value).strip()

    if not value:
        return False, None, "empty"

    if not HEX_PATTERN.fullmatch(value):
        return False, None, "non_hexadecimal"

    try:
        numeric_id = int(value, 16)
    except ValueError:
        return False, None, "conversion_error"

    if numeric_id > MAX_CAN_ID:
        return False, numeric_id, "out_of_29_bit_range"

    return True, numeric_id, ""

def inspect_payload(value):
    """
    Return payload validity and length information.

    Empty payloads are counted separately. They are not
    automatically classified as invalid because a zero-DLC
    CAN frame can be legitimate.
    """

    value = str(value).strip()

    if value == "":
        return True, 0, "empty"

    if not re.fullmatch(r"[0-9a-fA-F]+", value):
        return False, None, "non_hexadecimal"

    if len(value) % 2 != 0:
        return False, None, "odd_hex_character_count"

    length = len(value) // 2

    if length > 8:
        return False, length, "over_8_bytes"

    return True, length, ""

def clean_text_series(series):
    return series.astype(str).str.strip()

# ============================================================
# 3. FILE AUDIT
# ============================================================

def audit_file(file_path):

    relative_path = file_path.relative_to(DATASET_DIR)
    partition = relative_path.parts[0]

    result = {
        "file": str(relative_path),
        "partition": partition,
        "rows": 0,

        "blank_timestamp": 0,
        "blank_arbitration_id": 0,
        "blank_data_field": 0,
        "blank_attack": 0,

        "invalid_timestamp": 0,
        "nonfinite_timestamp": 0,
        "timestamp_decreases": 0,

        "invalid_arbitration_id": 0,
        "out_of_range_arbitration_id": 0,

        "invalid_payload": 0,
        "empty_payload": 0,
        "odd_length_payload": 0,
        "oversized_payload": 0,

        "invalid_labels": 0,
        "duplicate_rows_within_chunks": 0,
        "duplicate_adjacent_chunk_boundary": 0,
        "conflicting_label_groups": 0,

        "schema_valid": False,
        "status": "OK",
        "error": "",
    }

    payload_lengths = Counter()
    id_frequency = Counter()
    label_counts = Counter()
    payload_error_counts = Counter()

    previous_last_hash = None
    previous_last_timestamp = None

    try:

        reader = pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False,
            on_bad_lines="error",
            encoding="utf-8",
        )

        first_chunk = True

        for chunk_number, chunk in enumerate(reader, start=1):

            if first_chunk:
                result["schema_valid"] = (
                    list(chunk.columns) == EXPECTED_COLUMNS
                )

                if not result["schema_valid"]:
                    result["status"] = "SCHEMA_MISMATCH"
                    raise ValueError(
                        f"Unexpected columns: {list(chunk.columns)}"
                    )

                first_chunk = False

            chunk = chunk[EXPECTED_COLUMNS].copy()
            result["rows"] += len(chunk)

            # ------------------------------------------------
            # Missing / blank values
            # ------------------------------------------------

            for column in EXPECTED_COLUMNS:
                blanks = clean_text_series(chunk[column]).eq("")
                result[f"blank_{column}"] += int(blanks.sum())

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            labels = pd.to_numeric(
                chunk["attack"],
                errors="coerce"
            )

            valid_labels = labels.isin([0, 1])

            result["invalid_labels"] += int(
                (~valid_labels).sum()
            )

            label_counts.update(
                labels[valid_labels].astype(int).value_counts().to_dict()
            )

            # ------------------------------------------------
            # Timestamps
            # ------------------------------------------------

            timestamps = pd.to_numeric(
                chunk["timestamp"],
                errors="coerce"
            )

            finite = np.isfinite(
                timestamps.to_numpy(dtype=float)
            )

            result["invalid_timestamp"] += int(
                timestamps.isna().sum()
            )

            result["nonfinite_timestamp"] += int(
                (~timestamps.isna().to_numpy() & ~finite).sum()
            )

            valid_time = timestamps[finite]

            if len(valid_time) > 1:
                decreases = (
                    valid_time.diff().dropna() < 0
                ).sum()

                result["timestamp_decreases"] += int(decreases)

            if len(valid_time) > 0:
                first_timestamp = float(valid_time.iloc[0])

                if previous_last_timestamp is not None:
                    if first_timestamp < previous_last_timestamp:
                        result["timestamp_decreases"] += 1

                previous_last_timestamp = float(valid_time.iloc[-1])

            # ------------------------------------------------
            # Arbitration IDs
            # ------------------------------------------------

            ids = clean_text_series(chunk["arbitration_id"])

            for value in ids:
                valid, numeric_id, reason = parse_can_id(value)

                if not valid:
                    result["invalid_arbitration_id"] += 1

                    if reason == "out_of_29_bit_range":
                        result["out_of_range_arbitration_id"] += 1
                else:
                    id_frequency[numeric_id] += 1

            # ------------------------------------------------
            # Payloads
            # ------------------------------------------------

            payloads = clean_text_series(chunk["data_field"])

            for value in payloads:
                valid, length, reason = inspect_payload(value)

                if not valid:
                    result["invalid_payload"] += 1
                    payload_error_counts[reason] += 1
                    continue

                payload_lengths[length] += 1

                if reason == "empty":
                    result["empty_payload"] += 1

                if reason == "odd_hex_character_count":
                    result["odd_length_payload"] += 1

                if reason == "over_8_bytes":
                    result["oversized_payload"] += 1

            # ------------------------------------------------
            # Exact duplicate rows within this chunk
            # ------------------------------------------------

            hashes = pd.util.hash_pandas_object(
                chunk,
                index=False
            )

            result["duplicate_rows_within_chunks"] += int(
                hashes.duplicated().sum()
            )

            # Adjacent boundary duplicate check
            if len(hashes) > 0:
                first_hash = int(hashes.iloc[0])

                if (
                    previous_last_hash is not None
                    and first_hash == previous_last_hash
                ):
                    result["duplicate_adjacent_chunk_boundary"] += 1

                previous_last_hash = int(hashes.iloc[-1])

            # ------------------------------------------------
            # Conflicting labels within this chunk
            # ------------------------------------------------

            valid_label_frame = chunk.loc[valid_labels].copy()
            valid_label_frame["_label"] = (
                labels.loc[valid_labels].astype(int)
            )

            if not valid_label_frame.empty:

                conflict_groups = (
                    valid_label_frame.groupby(
                        [
                            "timestamp",
                            "arbitration_id",
                            "data_field",
                        ],
                        dropna=False
                    )["_label"]
                    .nunique()
                )

                result["conflicting_label_groups"] += int(
                    (conflict_groups > 1).sum()
                )

            print(
                f"    Chunk {chunk_number}: "
                f"{len(chunk):,} rows audited"
            )

    except Exception as error:

        if result["status"] != "SCHEMA_MISMATCH":
            result["status"] = "READ_ERROR"

        result["error"] = str(error)

    # --------------------------------------------------------
    # Save per-file distributions
    # --------------------------------------------------------

    payload_rows = [
        {
            "file": str(relative_path),
            "payload_length_bytes": length,
            "records": count,
        }
        for length, count in sorted(payload_lengths.items())
    ]

    id_rows = [
        {
            "file": str(relative_path),
            "arbitration_id_numeric": can_id,
            "records": count,
        }
        for can_id, count in id_frequency.items()
    ]

    label_rows = [
        {
            "file": str(relative_path),
            "label": label,
            "records": count,
        }
        for label, count in label_counts.items()
    ]

    error_rows = [
        {
            "file": str(relative_path),
            "payload_error": reason,
            "records": count,
        }
        for reason, count in payload_error_counts.items()
    ]

    return result, payload_rows, id_rows, label_rows, error_rows

# ============================================================
# 4. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CAN_BUS_AI — ORIGINAL DATA QUALITY AUDIT")
    print("=" * 70)

    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    files = sorted(DATASET_DIR.rglob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_payload_rows = []
    all_id_rows = []
    all_label_rows = []
    all_error_rows = []

    for index, file_path in enumerate(files, start=1):

        print(f"\n[{index}/{len(files)}] {file_path.name}")

        (
            result,
            payload_rows,
            id_rows,
            label_rows,
            error_rows,
        ) = audit_file(file_path)

        all_results.append(result)
        all_payload_rows.extend(payload_rows)
        all_id_rows.extend(id_rows)
        all_label_rows.extend(label_rows)
        all_error_rows.extend(error_rows)

        print(
            f"    Records: {result['rows']:,} | "
            f"Status: {result['status']}"
        )

    # --------------------------------------------------------
    # OUTPUT REPORTS
    # --------------------------------------------------------

    audit = pd.DataFrame(all_results)

    audit.to_csv(
        OUTPUT_DIR / "01_file_quality_summary.csv",
        index=False
    )

    pd.DataFrame(all_payload_rows).to_csv(
        OUTPUT_DIR / "02_payload_length_distribution.csv",
        index=False
    )

    pd.DataFrame(all_id_rows).to_csv(
        OUTPUT_DIR / "03_arbitration_id_frequency.csv",
        index=False
    )

    pd.DataFrame(all_label_rows).to_csv(
        OUTPUT_DIR / "04_file_level_label_distribution.csv",
        index=False
    )

    pd.DataFrame(all_error_rows).to_csv(
        OUTPUT_DIR / "05_payload_errors.csv",
        index=False
    )

    # --------------------------------------------------------
    # PARTITION SUMMARY
    # --------------------------------------------------------

    partition_summary = (
        audit.groupby("partition")
        .agg(
            files=("file", "count"),
            records=("rows", "sum"),
            invalid_labels=("invalid_labels", "sum"),
            invalid_timestamps=("invalid_timestamp", "sum"),
            nonfinite_timestamps=("nonfinite_timestamp", "sum"),
            invalid_ids=("invalid_arbitration_id", "sum"),
            invalid_payloads=("invalid_payload", "sum"),
            duplicate_rows=("duplicate_rows_within_chunks", "sum"),
            boundary_duplicates=(
                "duplicate_adjacent_chunk_boundary", "sum"
            ),
            conflicting_label_groups=(
                "conflicting_label_groups", "sum"
            ),
        )
        .reset_index()
    )

    partition_summary.to_csv(
        OUTPUT_DIR / "06_partition_quality_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # AUDIT SUMMARY
    # --------------------------------------------------------

    totals = audit.select_dtypes(include="number").sum()

    summary = [
        "# Original set_01 — Data Quality Audit",
        "",
        "## Scope",
        "",
        f"- Files inspected: {len(files):,}",
        f"- Records inspected: {int(totals['rows']):,}",
        f"- Chunk size: {CHUNK_SIZE:,}",
        "- Source CSV files were read-only.",
        "",
        "## Findings",
        "",
    ]

    fields = [
        ("Blank timestamps", "blank_timestamp"),
        ("Blank arbitration IDs", "blank_arbitration_id"),
        ("Blank payload fields", "blank_data_field"),
        ("Blank labels", "blank_attack"),
        ("Invalid timestamps", "invalid_timestamp"),
        ("Non-finite timestamps", "nonfinite_timestamp"),
        ("Timestamp decreases", "timestamp_decreases"),
        ("Invalid arbitration IDs", "invalid_arbitration_id"),
        ("Out-of-range CAN IDs", "out_of_range_arbitration_id"),
        ("Invalid payloads", "invalid_payload"),
        ("Empty payloads", "empty_payload"),
        ("Odd-length payloads", "odd_length_payload"),
        ("Oversized payloads", "oversized_payload"),
        ("Invalid labels", "invalid_labels"),
        ("Within-chunk duplicate rows", "duplicate_rows_within_chunks"),
        (
            "Adjacent chunk-boundary duplicates",
            "duplicate_adjacent_chunk_boundary",
        ),
        ("Conflicting label groups", "conflicting_label_groups"),
    ]

    for title, column in fields:
        summary.append(
            f"- {title}: {int(totals[column]):,}"
        )

    summary.extend([
        "",
        "## Duplicate detection limitations",
        "",
        "Exact duplicates are checked within individual chunks "
        "and at immediately adjacent chunk boundaries.",
        "This does not detect all possible duplicates across "
        "distant chunks or separate files.",
        "",
        "## Interpretation cautions",
        "",
        "- Empty payloads are counted separately, not automatically "
        "classified as invalid.",
        "- CAN IDs are checked against the 29-bit maximum; this does "
        "not establish whether a frame is standard or extended.",
        "- Timestamp decreases are not automatically treated as "
        "corruption.",
        "- Repeated CAN messages are not necessarily duplicate records.",
        "- No source records were modified.",
        "",
        "## Generated reports",
        "",
        "- 01_file_quality_summary.csv",
        "- 02_payload_length_distribution.csv",
        "- 03_arbitration_id_frequency.csv",
        "- 04_file_level_label_distribution.csv",
        "- 05_payload_errors.csv",
        "- 06_partition_quality_summary.csv",
    ])

    (OUTPUT_DIR / "audit_summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA QUALITY AUDIT COMPLETE")
    print("=" * 70)

    print(f"Files inspected: {len(files):,}")
    print(f"Records inspected: {int(totals['rows']):,}")
    print(f"Invalid timestamps: {int(totals['invalid_timestamp']):,}")
    print(f"Invalid IDs: {int(totals['invalid_arbitration_id']):,}")
    print(f"Invalid payloads: {int(totals['invalid_payload']):,}")
    print(f"Invalid labels: {int(totals['invalid_labels']):,}")
    print(
        "Within-chunk duplicates: "
        f"{int(totals['duplicate_rows_within_chunks']):,}"
    )

    print(f"\nReports saved to:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    main()