
from pathlib import Path
import pandas as pd
import numpy as np

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
GAP_THRESHOLD_SECONDS = 1.0
CONTEXT_ROWS = 3


def investigate_file(file_path):
    partition = file_path.parent.name
    file_name = file_path.name

    print(f"Checking {partition}/{file_name}")

    previous_rows = []
    row_number = 0
    findings = []

    for chunk in pd.read_csv(
        file_path,
        usecols=["timestamp", "arbitration_id", "data_field", "attack"],
        dtype={
            "timestamp": "float64",
            "arbitration_id": "string",
            "data_field": "string",
            "attack": "int8",
        },
        chunksize=CHUNK_SIZE,
    ):

        records = chunk.to_dict("records")

        for record in records:
            current = {
                **record,
                "source_row": row_number,
            }

            if previous_rows:
                previous = previous_rows[-1]
                delta = (
                    float(current["timestamp"])
                    - float(previous["timestamp"])
                )

                if abs(delta) > GAP_THRESHOLD_SECONDS:
                    findings.append({
                        "partition": partition,
                        "file": file_name,
                        "previous_row": previous["source_row"],
                        "current_row": current["source_row"],
                        "previous_timestamp": previous["timestamp"],
                        "current_timestamp": current["timestamp"],
                        "interval_seconds": delta,
                        "previous_attack": previous["attack"],
                        "current_attack": current["attack"],
                        "previous_id": previous["arbitration_id"],
                        "current_id": current["arbitration_id"],
                        "previous_payload": previous["data_field"],
                        "current_payload": current["data_field"],
                    })

            previous_rows.append(current)

            if len(previous_rows) > CONTEXT_ROWS + 1:
                previous_rows.pop(0)

            row_number += 1

    return findings


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATASET_ROOT.rglob("*.csv"))
    all_findings = []

    for file_path in files:
        all_findings.extend(investigate_file(file_path))

    columns = [
        "partition",
        "file",
        "previous_row",
        "current_row",
        "previous_timestamp",
        "current_timestamp",
        "interval_seconds",
        "previous_attack",
        "current_attack",
        "previous_id",
        "current_id",
        "previous_payload",
        "current_payload",
    ]

    result = pd.DataFrame(all_findings, columns=columns)

    if not result.empty:
        result = result.sort_values(
            "interval_seconds",
            key=lambda s: s.abs(),
            ascending=False,
        )

    result.to_csv(
        OUTPUT_DIR / "06_timestamp_gaps_over_1_second.csv",
        index=False,
    )

    print("\nInvestigation complete.")
    print(f"Gaps found: {len(result):,}")
    print(
        "Output:",
        OUTPUT_DIR / "06_timestamp_gaps_over_1_second.csv",
    )

    if not result.empty:
        print("\nLargest gaps:")
        print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
