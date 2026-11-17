
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "set_01"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_1"
    / "can_traffic_analysis"
)

CHUNK_SIZE = 200_000

EXPECTED_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]

HEX_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{2})*$")


# ============================================================
# HELPERS
# ============================================================

def parse_arbitration_id(value):
    """Interpret arbitration IDs as hexadecimal values."""
    try:
        value = str(value).strip()
        if value.lower().startswith("0x"):
            value = value[2:]
        return int(value, 16)
    except (ValueError, TypeError):
        return None


def parse_payload(value):
    """
    Convert a hexadecimal payload into bytes.
    Empty payloads are retained as zero-length payloads.
    Invalid payloads return None.
    """
    value = str(value).strip()

    if value == "":
        return b""

    if not HEX_PATTERN.fullmatch(value):
        return None

    try:
        return bytes.fromhex(value)
    except ValueError:
        return None


def get_partition(file_path):
    """Return the partition directory immediately above the CSV."""
    return file_path.parent.name


def safe_rate(numerator, denominator):
    return (
        numerator / denominator
        if denominator
        else 0.0
    )


# ============================================================
# AGGREGATE STORAGE
# ============================================================

file_stats = defaultdict(Counter)
partition_stats = defaultdict(Counter)

# (file, arbitration ID) -> normal/attack counts
file_id_counts = defaultdict(Counter)

# (partition, arbitration ID) -> normal/attack counts
partition_id_counts = defaultdict(Counter)

# (partition, attack label, payload length) -> count
payload_length_counts = Counter()

# (partition, attack label, byte position, byte value) -> count
byte_value_counts = Counter()

# Data validation counters
quality_counts = defaultdict(Counter)


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_file(file_path):
    file_name = file_path.name
    partition = get_partition(file_path)

    print(f"Processing: {partition}/{file_name}")

    file_id_set = set()
    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=EXPECTED_COLUMNS,
        dtype={
            "timestamp": "float64",
            "arbitration_id": "string",
            "data_field": "string",
            "attack": "int8",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        labels = chunk["attack"]

        normal_count = int((labels == 0).sum())
        attack_count = int((labels == 1).sum())

        file_stats[(partition, file_name)]["records"] += len(chunk)
        file_stats[(partition, file_name)]["normal"] += normal_count
        file_stats[(partition, file_name)]["attack"] += attack_count

        partition_stats[partition]["records"] += len(chunk)
        partition_stats[partition]["normal"] += normal_count
        partition_stats[partition]["attack"] += attack_count

        # ----------------------------------------------------
        # Arbitration IDs
        # ----------------------------------------------------

        parsed_ids = chunk["arbitration_id"].map(
            parse_arbitration_id
        )

        valid_id_mask = parsed_ids.notna()
        valid_ids = parsed_ids[valid_id_mask].astype("int64")

        file_stats[(partition, file_name)]["invalid_ids"] += int(
            (~valid_id_mask).sum()
        )

        file_id_set.update(valid_ids.unique().tolist())

        file_stats[(partition, file_name)]["id_11bit_rows"] += int(
            ((valid_ids >= 0) & (valid_ids <= 0x7FF)).sum()
        )

        file_stats[(partition, file_name)]["id_above_11bit_rows"] += int(
            (valid_ids > 0x7FF).sum()
        )

        # Count normal and attack records per ID.
        id_frame = pd.DataFrame({
            "arbitration_id": valid_ids,
            "attack": labels.loc[valid_id_mask].to_numpy(),
        })

        grouped_ids = (
            id_frame.groupby(["arbitration_id", "attack"])
            .size()
        )

        for (arb_id, attack_label), count in grouped_ids.items():
            key = (partition, file_name, int(arb_id))
            label_name = "attack" if attack_label == 1 else "normal"
            file_id_counts[key][label_name] += int(count)

            partition_key = (partition, int(arb_id))
            partition_id_counts[partition_key][label_name] += int(count)


        # ----------------------------------------------------
        # Payload lengths and byte-value distributions
        # Vectorised to avoid a Python loop over every record.
        # ----------------------------------------------------

        payloads = chunk["data_field"].fillna("").str.strip()
        valid_payload_mask = payloads.str.fullmatch(
            r"(?:[0-9a-fA-F]{2})*",
            na=False,
        )

        quality_counts[(partition, file_name)]["invalid_payloads"] += int(
            (~valid_payload_mask).sum()
        )
        
        quality_counts[(partition, file_name)]["rows_processed"] += len(chunk)
        
        valid_payloads = payloads[valid_payload_mask]
        valid_labels = labels.loc[valid_payload_mask]

        payload_lengths = valid_payloads.str.len() // 2

        length_frame = pd.DataFrame({
            "length": payload_lengths,
            "attack": valid_labels,
        })

        for (length, attack_label), count in (
            length_frame.groupby(["length", "attack"]).size().items()
        ):
            label_name = "attack" if attack_label == 1 else "normal"
            payload_length_counts[
                (partition, label_name, int(length))
            ] += int(count)

        # Analyse each of the eight possible CAN data bytes.
        # The original audit found no payloads longer than 8 bytes.
        for position in range(8):
            start = position * 2
            end = start + 2

            byte_strings = valid_payloads.str.slice(start, end)

            present_mask = byte_strings.str.len() == 2

            if not present_mask.any():
                continue

            byte_frame = pd.DataFrame({
                "byte_value": byte_strings[present_mask].str.upper(),
                "attack": valid_labels.loc[present_mask],
            })

            grouped_bytes = (
                byte_frame.groupby(["attack", "byte_value"])
                .size()
            )

            for (attack_label, byte_hex), count in grouped_bytes.items():
                byte_value = int(byte_hex, 16)
                label_name = "attack" if attack_label == 1 else "normal"

                byte_value_counts[
                    (partition, label_name, position, byte_value)
                ] += int(count)



# ============================================================
# OUTPUT GENERATION
# ============================================================

def write_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_ids_by_file = Counter()

    for partition, file_name, arbitration_id in file_id_counts:
        unique_ids_by_file[(partition, file_name)] += 1

    # --------------------------------------------------------
    # 1. File-level summary
    # --------------------------------------------------------

    file_rows = []

    for (partition, file_name), stats in file_stats.items():
        records = stats["records"]
        file_rows.append({
            "partition": partition,
            "file": file_name,
            "records": records,
            "normal": stats["normal"],
            "attack": stats["attack"],
            "attack_percentage": 100 * safe_rate(
                stats["attack"], records
            ),
            "unique_arbitration_ids": unique_ids_by_file[(partition, file_name)],
            "11bit_range_rows": stats["id_11bit_rows"],
            "above_11bit_range_rows": stats["id_above_11bit_rows"],
            "invalid_ids": stats["invalid_ids"],
        })

    pd.DataFrame(file_rows).sort_values(
        ["partition", "file"]
    ).to_csv(
        OUTPUT_DIR / "01_file_traffic_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Partition-level summary
    # --------------------------------------------------------

    partition_rows = []

    for partition, stats in partition_stats.items():
        records = stats["records"]

        partition_rows.append({
            "partition": partition,
            "records": records,
            "normal": stats["normal"],
            "attack": stats["attack"],
            "attack_percentage": 100 * safe_rate(
                stats["attack"], records
            ),
        })

    pd.DataFrame(partition_rows).sort_values(
        "partition"
    ).to_csv(
        OUTPUT_DIR / "02_partition_traffic_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Arbitration ID frequency and attack association
    # --------------------------------------------------------

    file_id_rows = []

    for (partition, file_name, arb_id), counts in file_id_counts.items():
        total = counts["normal"] + counts["attack"]

        file_id_rows.append({
            "partition": partition,
            "file": file_name,
            "arbitration_id_hex": f"0x{arb_id:X}",
            "arbitration_id_decimal": arb_id,
            "normal_count": counts["normal"],
            "attack_count": counts["attack"],
            "total_count": total,
            "attack_percentage": 100 * safe_rate(
                counts["attack"], total
            ),
        })

    file_id_df = pd.DataFrame(file_id_rows)

    if not file_id_df.empty:
        file_id_df.sort_values(
            ["partition", "file", "total_count"],
            ascending=[True, True, False],
        ).to_csv(
            OUTPUT_DIR / "03_file_arbitration_id_analysis.csv",
            index=False,
        )

    partition_id_rows = []

    for (partition, arb_id), counts in partition_id_counts.items():
        total = counts["normal"] + counts["attack"]

        partition_id_rows.append({
            "partition": partition,
            "arbitration_id_hex": f"0x{arb_id:X}",
            "arbitration_id_decimal": arb_id,
            "normal_count": counts["normal"],
            "attack_count": counts["attack"],
            "total_count": total,
            "attack_percentage": 100 * safe_rate(
                counts["attack"], total
            ),
        })

    partition_id_df = pd.DataFrame(partition_id_rows)

    if not partition_id_df.empty:
        partition_id_df.sort_values(
            ["partition", "total_count"],
            ascending=[True, False],
        ).to_csv(
            OUTPUT_DIR / "04_partition_arbitration_id_analysis.csv",
            index=False,
        )

    # --------------------------------------------------------
    # 4. Payload length distribution
    # --------------------------------------------------------

    payload_length_rows = []

    for (partition, label, length), count in payload_length_counts.items():
        payload_length_rows.append({
            "partition": partition,
            "class": label,
            "payload_length_bytes": length,
            "record_count": count,
        })

    pd.DataFrame(payload_length_rows).sort_values(
        ["partition", "class", "payload_length_bytes"]
    ).to_csv(
        OUTPUT_DIR / "05_payload_length_by_class.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Payload byte-value distribution
    # --------------------------------------------------------

    byte_rows = []

    for (
        partition,
        label,
        position,
        byte_value,
    ), count in byte_value_counts.items():

        byte_rows.append({
            "partition": partition,
            "class": label,
            "byte_position": position,
            "byte_value_hex": f"0x{byte_value:02X}",
            "byte_value_decimal": byte_value,
            "count": count,
        })

    pd.DataFrame(byte_rows).sort_values(
        ["partition", "class", "byte_position", "byte_value_decimal"]
    ).to_csv(
        OUTPUT_DIR / "06_payload_byte_value_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Processing quality summary
    # --------------------------------------------------------

    quality_rows = []

    for (partition, file_name), stats in quality_counts.items():
        quality_rows.append({
            "partition": partition,
            "file": file_name,
            "rows_processed": stats["rows_processed"],
            "invalid_payloads": stats["invalid_payloads"],
        })

    pd.DataFrame(quality_rows).sort_values(
        ["partition", "file"]
    ).to_csv(
        OUTPUT_DIR / "07_processing_quality.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Human-readable summary
    # --------------------------------------------------------

    total_records = sum(
        stats["records"] for stats in partition_stats.values()
    )

    total_attack = sum(
        stats["attack"] for stats in partition_stats.values()
    )

    total_normal = sum(
        stats["normal"] for stats in partition_stats.values()
    )

    total_invalid_payloads = sum(
        stats["invalid_payloads"] for stats in quality_counts.values()
    )

    summary = f"""# CAN Traffic Analysis — Set 01

## Scope

- Source: Original Set 01 CSV files
- Processing: Chunk-based, read-only
- Chunk size: {CHUNK_SIZE:,}
- Total records processed: {total_records:,}

## Class distribution

- Normal records: {total_normal:,}
- Attack records: {total_attack:,}
- Attack prevalence: {100 * safe_rate(total_attack, total_records):.6f}%

## Payload validation

- Invalid payloads encountered during this analysis: {total_invalid_payloads:,}
- Empty payloads are retained as zero-length payloads.
- Payloads are not padded or modified.

## Arbitration ID interpretation

IDs are interpreted as hexadecimal values. The analysis reports whether
their numerical values fall within the 11-bit range (0–2047) or above it.
This is a numerical range check, not proof of the actual CAN frame format.

## Outputs

1. `01_file_traffic_summary.csv`
2. `02_partition_traffic_summary.csv`
3. `03_file_arbitration_id_analysis.csv`
4. `04_partition_arbitration_id_analysis.csv`
5. `05_payload_length_by_class.csv`
6. `06_payload_byte_value_distribution.csv`
7. `07_processing_quality.csv`

## Interpretation cautions

- An ID with a high attack percentage is not necessarily a reliable
  intrusion indicator; its distribution may be specific to one file,
  vehicle, or attack scenario.
- Byte-level differences are descriptive associations, not proof of
  causal relationships.
- The binary attack label does not independently identify the attack type.
- The results must be interpreted alongside the official partition
  structure and the later temporal analysis.
"""

    (OUTPUT_DIR / "analysis_summary.md").write_text(
        summary,
        encoding="utf-8",
    )

    print("\nOutput files written to:")
    print(OUTPUT_DIR)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    csv_files = sorted(DATASET_ROOT.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under: {DATASET_ROOT}"
        )

    print("=" * 65)
    print("CAN BUS TRAFFIC ANALYSIS — SET 01")
    print("=" * 65)
    print(f"CSV files discovered: {len(csv_files)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print()

    for file_path in csv_files:
        process_file(file_path)

    write_outputs()

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
