
from pathlib import Path
import json
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_ROOT = PROJECT_ROOT / "Dataset" / "set_01"
SAMPLED_ROOT = PROJECT_ROOT / "data" / "sampled_stratified" / "set_01"

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_2"
    / "stratified_sampling"
    / "validation"
)

FILE_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_2"
    / "stratified_sampling"
    / "01_file_sampling_summary.csv"
)

CHUNK_SIZE = 200_000

EXPECTED_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def read_csv_chunks(path):
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )


def normalise_id(value):
    value = str(value).strip().lower()

    if value.startswith("0x"):
        value = value[2:]

    return value.lstrip("0") or "0"


def payload_length(value):
    value = str(value).strip()

    if value == "":
        return 0

    return len(value) // 2


def inspect_file(path):
    """Collect row, class, ID and payload-length statistics."""

    total_rows = 0
    normal_count = 0
    attack_count = 0
    invalid_labels = 0
    invalid_ids = 0
    invalid_payloads = 0

    id_counts = {}
    payload_counts = {}

    observed_columns = None

    for chunk in read_csv_chunks(path):
        if observed_columns is None:
            observed_columns = list(chunk.columns)

        total_rows += len(chunk)

        labels = chunk["attack"].astype(str).str.strip()

        normal_count += int((labels == "0").sum())
        attack_count += int((labels == "1").sum())
        invalid_labels += int((~labels.isin(["0", "1"])).sum())

        for value in chunk["arbitration_id"]:
            try:
                normalised = normalise_id(value)
                int(normalised, 16)
                id_counts[normalised] = id_counts.get(normalised, 0) + 1
            except (ValueError, TypeError):
                invalid_ids += 1

        for value in chunk["data_field"]:
            value = str(value).strip()

            if value == "":
                length = 0
            else:
                try:
                    int(value, 16)
                    if len(value) % 2 != 0:
                        raise ValueError("Odd-length payload")
                    length = len(value) // 2
                    if length > 8:
                        raise ValueError("Payload exceeds 8 bytes")
                except ValueError:
                    invalid_payloads += 1
                    continue

            payload_counts[length] = payload_counts.get(length, 0) + 1

    return {
        "rows": total_rows,
        "normal": normal_count,
        "attack": attack_count,
        "invalid_labels": invalid_labels,
        "invalid_ids": invalid_ids,
        "invalid_payloads": invalid_payloads,
        "columns": observed_columns,
        "id_counts": id_counts,
        "payload_counts": payload_counts,
    }


def distribution_difference(original, sampled):
    """Calculate total variation distance between distributions."""

    keys = set(original) | set(sampled)

    original_total = sum(original.values())
    sampled_total = sum(sampled.values())

    if original_total == 0 or sampled_total == 0:
        return None

    difference = sum(
        abs(
            original.get(key, 0) / original_total
            - sampled.get(key, 0) / sampled_total
        )
        for key in keys
    ) / 2

    return difference


# ---------------------------------------------------------
# Main validation
# ---------------------------------------------------------

def main():

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    if not FILE_SUMMARY.exists():
        raise FileNotFoundError(
            f"Sampling summary not found: {FILE_SUMMARY}"
        )

    summary = pd.read_csv(FILE_SUMMARY)

    results = []

    for _, item in summary.iterrows():

        partition = item["partition"]
        filename = item["file"]

        original_path = ORIGINAL_ROOT / partition / filename
        sampled_path = SAMPLED_ROOT / partition / filename

        print(f"\nValidating: {partition}/{filename}")

        if not original_path.exists():
            print("  ERROR: Original file missing")
            results.append({
                "partition": partition,
                "file": filename,
                "status": "FAIL",
                "reason": "Original file missing",
            })
            continue

        if not sampled_path.exists():
            print("  ERROR: Sampled file missing")
            results.append({
                "partition": partition,
                "file": filename,
                "status": "FAIL",
                "reason": "Sampled file missing",
            })
            continue

        original = inspect_file(original_path)
        sampled = inspect_file(sampled_path)

        expected_rows = int(item["sampled_records"])
        expected_normal = int(item["sampled_normal"])
        expected_attack = int(item["sampled_attack"])

        checks = {
            "schema_match": (
                original["columns"] == EXPECTED_COLUMNS
                and sampled["columns"] == EXPECTED_COLUMNS
            ),
            "sampled_row_count": (
                sampled["rows"] == expected_rows
            ),
            "normal_count": (
                sampled["normal"] == expected_normal
            ),
            "attack_count": (
                sampled["attack"] == expected_attack
            ),
            "labels_valid": (
                original["invalid_labels"] == 0
                and sampled["invalid_labels"] == 0
            ),
            "ids_valid": (
                original["invalid_ids"] == 0
                and sampled["invalid_ids"] == 0
            ),
            "payloads_valid": (
                original["invalid_payloads"] == 0
                and sampled["invalid_payloads"] == 0
            ),
            "class_counts_consistent": (
                sampled["normal"] + sampled["attack"]
                == sampled["rows"]
            ),
            "sampled_not_larger": (
                sampled["rows"] <= original["rows"]
            ),
        }

        id_distance = distribution_difference(
            original["id_counts"],
            sampled["id_counts"],
        )

        payload_distance = distribution_difference(
            original["payload_counts"],
            sampled["payload_counts"],
        )

        all_passed = all(checks.values())

        result = {
            "partition": partition,
            "file": filename,
            "status": "PASS" if all_passed else "FAIL",
            "original_rows": original["rows"],
            "sampled_rows": sampled["rows"],
            "actual_retention_percentage": (
                100 * sampled["rows"] / original["rows"]
                if original["rows"] else 0
            ),
            "original_normal": original["normal"],
            "sampled_normal": sampled["normal"],
            "original_attack": original["attack"],
            "sampled_attack": sampled["attack"],
            "original_attack_percentage": (
                100 * original["attack"] / original["rows"]
                if original["rows"] else 0
            ),
            "sampled_attack_percentage": (
                100 * sampled["attack"] / sampled["rows"]
                if sampled["rows"] else 0
            ),
            "arbitration_id_distribution_distance": id_distance,
            "payload_length_distribution_distance": payload_distance,
            **checks,
        }

        results.append(result)

        print(
            f"  Original: {original['rows']:,} | "
            f"Sampled: {sampled['rows']:,}"
        )
        print(
            f"  Attack %: "
            f"{result['original_attack_percentage']:.6f}% -> "
            f"{result['sampled_attack_percentage']:.6f}%"
        )
        print(
            f"  ID distribution distance: {id_distance:.6f}"
            if id_distance is not None else
            "  ID distribution distance: N/A"
        )
        print(
            f"  Payload distribution distance: {payload_distance:.6f}"
            if payload_distance is not None else
            "  Payload distribution distance: N/A"
        )
        print(f"  Status: {result['status']}")

    # -----------------------------------------------------
    # Save reports
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_ROOT / "01_file_validation_results.csv",
        index=False,
    )

    summary_rows = len(results_df)
    passed = int((results_df["status"] == "PASS").sum())
    failed = summary_rows - passed

    partition_summary = (
        results_df.groupby("partition", dropna=False)
        .agg(
            files=("file", "count"),
            passed=("status", lambda x: (x == "PASS").sum()),
            failed=("status", lambda x: (x == "FAIL").sum()),
            original_rows=("original_rows", "sum"),
            sampled_rows=("sampled_rows", "sum"),
        )
        .reset_index()
    )

    partition_summary.to_csv(
        REPORT_ROOT / "02_partition_validation_summary.csv",
        index=False,
    )

    report = {
        "files_checked": summary_rows,
        "files_passed": passed,
        "files_failed": failed,
        "overall_status": "PASS" if failed == 0 else "FAIL",
        "validation_scope": [
            "CSV schema",
            "sampled row counts",
            "normal and attack counts",
            "binary label validity",
            "arbitration ID validity",
            "payload validity",
            "arbitration ID distribution",
            "payload length distribution",
        ],
        "important_limitation": (
            "This script does not independently prove that sampled rows "
            "form an ordered subsequence of the original files. "
            "Source-order preservation is recorded in the sampler "
            "configuration and should be verified separately if required."
        ),
    }

    with open(
        REPORT_ROOT / "03_validation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 55)
    print("VALIDATION COMPLETE")
    print("=" * 55)
    print(f"Files checked: {summary_rows}")
    print(f"Passed:        {passed}")
    print(f"Failed:        {failed}")
    print(f"Overall:       {report['overall_status']}")
    print(f"\nReports saved to:\n{REPORT_ROOT}")


if __name__ == "__main__":
    main()
