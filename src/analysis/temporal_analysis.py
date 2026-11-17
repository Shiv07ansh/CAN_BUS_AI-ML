
from pathlib import Path
from collections import Counter, defaultdict
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
    / "stage_1"
    / "temporal_analysis"
)

CHUNK_SIZE = 200_000

# Retain one interval out of every N for quantile calculations.
# This keeps memory use manageable on the full dataset.
INTERVAL_SAMPLE_STEP = 100

# One-second windows, expressed in the timestamp's presumed
# unit of seconds.
WINDOW_SECONDS = 1


# ============================================================
# AGGREGATE STORAGE
# ============================================================

file_stats = defaultdict(Counter)
partition_stats = defaultdict(Counter)

# Sampled positive inter-arrival times, grouped by file and
# by the class label of the current message.
interval_samples = defaultdict(list)

# One-second message counts:
# (partition, filename, second_offset) -> normal/attack counts
window_counts = defaultdict(Counter)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path):
    partition = file_path.parent.name
    file_name = file_path.name
    key = (partition, file_name)

    print(f"Processing: {partition}/{file_name}")

    previous_timestamp = None
    first_timestamp = None
    last_timestamp = None

    interval_sample_counter = Counter()

    for chunk in pd.read_csv(
        file_path,
        usecols=["timestamp", "attack"],
        dtype={
            "timestamp": "float64",
            "attack": "int8",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        timestamps = chunk["timestamp"].to_numpy(dtype=np.float64)
        labels = chunk["attack"].to_numpy(dtype=np.int8)

        if len(timestamps) == 0:
            continue

        file_stats[key]["records"] += len(chunk)
        partition_stats[partition]["records"] += len(chunk)

        file_stats[key]["normal"] += int((labels == 0).sum())
        file_stats[key]["attack"] += int((labels == 1).sum())

        partition_stats[partition]["normal"] += int(
            (labels == 0).sum()
        )
        partition_stats[partition]["attack"] += int(
            (labels == 1).sum()
        )

        if first_timestamp is None:
            first_timestamp = float(timestamps[0])

        # ----------------------------------------------------
        # Inter-arrival times in original row order
        # ----------------------------------------------------

        if previous_timestamp is not None:
            all_timestamps = np.concatenate(
                ([previous_timestamp], timestamps)
            )
            interval_labels = labels
        else:
            all_timestamps = timestamps
            interval_labels = labels[1:]

        if previous_timestamp is not None:
            intervals = np.diff(all_timestamps)
        else:
            intervals = np.diff(timestamps)

        # The first interval in the first chunk has no
        # preceding message within this file.
        if len(intervals) > 0:
            file_stats[key]["intervals"] += len(intervals)
            partition_stats[partition]["intervals"] += len(intervals)

            negative_mask = intervals < 0
            zero_mask = intervals == 0
            positive_mask = intervals > 0

            file_stats[key]["negative_intervals"] += int(
                negative_mask.sum()
            )
            file_stats[key]["zero_intervals"] += int(
                zero_mask.sum()
            )
            file_stats[key]["positive_intervals"] += int(
                positive_mask.sum()
            )

            partition_stats[partition]["negative_intervals"] += int(
                negative_mask.sum()
            )
            partition_stats[partition]["zero_intervals"] += int(
                zero_mask.sum()
            )
            partition_stats[partition]["positive_intervals"] += int(
                positive_mask.sum()
            )

            # Assign each interval to the label of the current
            # message (the message arriving at the interval end).
            for class_label, class_name in [(0, "normal"), (1, "attack")]:
                class_mask = (interval_labels == class_label)
                class_intervals = intervals[class_mask]

                file_stats[key][f"{class_name}_intervals"] += len(
                    class_intervals
                )

                positive_class_intervals = class_intervals[
                    class_intervals > 0
                ]

                file_stats[key][
                    f"{class_name}_positive_intervals"
                ] += len(positive_class_intervals)

                # Deterministic systematic sample for quantiles.
                if len(positive_class_intervals):
                    sample = positive_class_intervals[
                        ::INTERVAL_SAMPLE_STEP
                    ]

                    interval_samples[
                        (partition, file_name, class_name)
                    ].extend(sample.tolist())

            file_stats[key]["max_positive_interval"] = max(
                file_stats[key]["max_positive_interval"],
                float(intervals[positive_mask].max())
                if positive_mask.any() else 0.0,
            )

        # Preserve last timestamp for the next chunk.
        previous_timestamp = float(timestamps[-1])
        last_timestamp = previous_timestamp

        # ----------------------------------------------------
        # One-second message-count windows
        # ----------------------------------------------------

        offsets = np.floor(
            (timestamps - first_timestamp) / WINDOW_SECONDS
        ).astype(np.int64)

        window_frame = pd.DataFrame({
            "second_offset": offsets,
            "attack": labels,
        })

        grouped = window_frame.groupby(
            ["second_offset", "attack"]
        ).size()

        for (second_offset, attack_label), count in grouped.items():
            window_key = (
                partition,
                file_name,
                int(second_offset),
            )

            class_name = "attack" if attack_label == 1 else "normal"
            window_counts[window_key][class_name] += int(count)

    # --------------------------------------------------------
    # File-level timestamp range and duration
    # --------------------------------------------------------

    if first_timestamp is not None and last_timestamp is not None:
        file_stats[key]["first_timestamp"] = first_timestamp
        file_stats[key]["last_timestamp"] = last_timestamp
        file_stats[key]["timestamp_span"] = (
            last_timestamp - first_timestamp
        )

    print(
        f"  Records: {file_stats[key]['records']:,} | "
        f"Negative intervals: "
        f"{file_stats[key]['negative_intervals']:,}"
    )


# ============================================================
# OUTPUT GENERATION
# ============================================================

def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def quantile_or_nan(values, q):
    if not values:
        return np.nan
    return float(np.quantile(values, q))


def write_outputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. File-level temporal summary
    # --------------------------------------------------------

    file_rows = []

    for (partition, file_name), stats in file_stats.items():
        span = stats["timestamp_span"]
        records = stats["records"]

        file_rows.append({
            "partition": partition,
            "file": file_name,
            "records": records,
            "normal": stats["normal"],
            "attack": stats["attack"],
            "first_timestamp": stats["first_timestamp"],
            "last_timestamp": stats["last_timestamp"],
            "timestamp_span_seconds": span,
            "average_messages_per_second": (
                (records - 1) / span if span > 0 else np.nan
            ),
            "intervals": stats["intervals"],
            "negative_intervals": stats["negative_intervals"],
            "zero_intervals": stats["zero_intervals"],
            "positive_intervals": stats["positive_intervals"],
            "negative_interval_percentage": (
                100 * safe_rate(
                    stats["negative_intervals"],
                    stats["intervals"],
                )
            ),
            "max_positive_interval_seconds": (
                stats["max_positive_interval"]
            ),
        })

    file_df = pd.DataFrame(file_rows).sort_values(
        ["partition", "file"]
    )

    file_df.to_csv(
        OUTPUT_DIR / "01_file_temporal_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Inter-arrival time statistics
    # --------------------------------------------------------

    interval_rows = []

    for (partition, file_name, class_name), values in (
        interval_samples.items()
    ):
        interval_rows.append({
            "partition": partition,
            "file": file_name,
            "class": class_name,
            "sampled_positive_intervals": len(values),
            "mean_seconds": (
                float(np.mean(values)) if values else np.nan
            ),
            "median_seconds": quantile_or_nan(values, 0.50),
            "p90_seconds": quantile_or_nan(values, 0.90),
            "p95_seconds": quantile_or_nan(values, 0.95),
            "p99_seconds": quantile_or_nan(values, 0.99),
            "max_sampled_seconds": (
                float(np.max(values)) if values else np.nan
            ),
        })

    pd.DataFrame(interval_rows).sort_values(
        ["partition", "file", "class"]
    ).to_csv(
        OUTPUT_DIR / "02_interarrival_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. One-second window distributions
    # --------------------------------------------------------

    window_rows = []

    for (partition, file_name, second_offset), counts in (
        window_counts.items()
    ):
        total = counts["normal"] + counts["attack"]

        window_rows.append({
            "partition": partition,
            "file": file_name,
            "second_offset": second_offset,
            "normal_messages": counts["normal"],
            "attack_messages": counts["attack"],
            "total_messages": total,
            "attack_percentage": (
                100 * safe_rate(counts["attack"], total)
            ),
        })

    window_df = pd.DataFrame(window_rows)

    if not window_df.empty:
        window_df.sort_values(
            ["partition", "file", "second_offset"]
        ).to_csv(
            OUTPUT_DIR / "03_one_second_message_windows.csv",
            index=False,
        )

    # --------------------------------------------------------
    # 4. Per-file message-rate statistics from windows
    # --------------------------------------------------------

    rate_rows = []

    if not window_df.empty:
        for (partition, file_name), group in window_df.groupby(
            ["partition", "file"]
        ):
            rates = group["total_messages"].to_numpy()

            rate_rows.append({
                "partition": partition,
                "file": file_name,
                "number_of_observed_windows": len(rates),
                "mean_messages_per_second": float(np.mean(rates)),
                "median_messages_per_second": float(np.median(rates)),
                "p95_messages_per_second": float(
                    np.quantile(rates, 0.95)
                ),
                "p99_messages_per_second": float(
                    np.quantile(rates, 0.99)
                ),
                "maximum_messages_in_one_second": int(
                    np.max(rates)
                ),
                "windows_with_attack_messages": int(
                    (group["attack_messages"] > 0).sum()
                ),
            })

    pd.DataFrame(rate_rows).sort_values(
        ["partition", "file"]
    ).to_csv(
        OUTPUT_DIR / "04_file_message_rate_statistics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Partition-level summary
    # --------------------------------------------------------

    partition_rows = []

    for partition, stats in partition_stats.items():
        partition_rows.append({
            "partition": partition,
            "records": stats["records"],
            "normal": stats["normal"],
            "attack": stats["attack"],
            "intervals": stats["intervals"],
            "negative_intervals": stats["negative_intervals"],
            "zero_intervals": stats["zero_intervals"],
            "positive_intervals": stats["positive_intervals"],
            "negative_interval_percentage": (
                100 * safe_rate(
                    stats["negative_intervals"],
                    stats["intervals"],
                )
            ),
        })

    pd.DataFrame(partition_rows).sort_values(
        "partition"
    ).to_csv(
        OUTPUT_DIR / "05_partition_temporal_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. Human-readable summary
    # --------------------------------------------------------

    total_records = sum(
        stats["records"] for stats in partition_stats.values()
    )

    total_negative = sum(
        stats["negative_intervals"]
        for stats in partition_stats.values()
    )

    total_zero = sum(
        stats["zero_intervals"]
        for stats in partition_stats.values()
    )

    total_positive = sum(
        stats["positive_intervals"]
        for stats in partition_stats.values()
    )

    summary = f"""# Temporal Analysis — Set 01

## Scope

- Original Set 01 CSV files
- Chunk-based processing
- Chunk size: {CHUNK_SIZE:,}
- Inter-arrival times calculated within each file
- Original row order preserved
- One-second message-count windows

## Records

- Total records: {total_records:,}

## Timestamp ordering

- Negative inter-arrival intervals: {total_negative:,}
- Zero inter-arrival intervals: {total_zero:,}
- Positive inter-arrival intervals: {total_positive:,}

## Method

Inter-arrival time is calculated as the difference between the
timestamp of the current message and the preceding message in
the original file order.

Intervals are associated with the class label of the current
message. Only positive intervals are included in the sampled
inter-arrival quantile statistics.

The quantile calculations use a deterministic systematic sample:
one interval is retained for every {INTERVAL_SAMPLE_STEP} positive
intervals.

One-second windows are anchored to the first timestamp in each
file. They are used to describe message counts, not to reorder
the original records.

## Outputs

1. `01_file_temporal_summary.csv`
2. `02_interarrival_statistics.csv`
3. `03_one_second_message_windows.csv`
4. `04_file_message_rate_statistics.csv`
5. `05_partition_temporal_summary.csv`

## Interpretation cautions

- Timestamp reversals are counted, not corrected.
- Intervals are not calculated across separate files.
- Inter-arrival statistics are based on a systematic sample.
- A one-second window may contain records whose timestamps
  appeared out of order in the original file.
- Attack-labelled message timing does not independently define
  attack episode boundaries.
- Message rates depend on the timestamp unit being seconds.
"""

    (OUTPUT_DIR / "analysis_summary.md").write_text(
        summary,
        encoding="utf-8",
    )

    print("\nReports saved to:")
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
    print("TEMPORAL ANALYSIS — SET 01")
    print("=" * 65)
    print(f"Files discovered: {len(csv_files)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print()

    for file_path in csv_files:
        process_file(file_path)

    write_outputs()

    print("\nTemporal analysis completed.")


if __name__ == "__main__":
    main()
