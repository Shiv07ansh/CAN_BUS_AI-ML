
import csv
import json
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_ROOT = PROJECT_ROOT / "Dataset" / "set_01"
SAMPLED_ROOT = (
    PROJECT_ROOT / "data" / "sampled_stratified" / "set_01"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_2"
    / "stratified_sampling"
    / "01_file_sampling_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_2"
    / "stratified_sampling"
    / "validation"
)

EXPECTED_HEADER = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

def verify_file(original_path, sampled_path):

    original_count = 0
    sampled_count = 0
    matched_count = 0

    with open(
        original_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as original_file, open(
        sampled_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as sampled_file:

        original_reader = csv.reader(original_file)
        sampled_reader = csv.reader(sampled_file)

        original_header = next(original_reader, None)
        sampled_header = next(sampled_reader, None)

        if original_header != EXPECTED_HEADER:
            return {
                "status": "FAIL",
                "reason": "Original header mismatch",
            }

        if sampled_header != EXPECTED_HEADER:
            return {
                "status": "FAIL",
                "reason": "Sampled header mismatch",
            }

        original_iterator = iter(original_reader)

        for sampled_row in sampled_reader:

            sampled_count += 1

            if len(sampled_row) != len(EXPECTED_HEADER):
                return {
                    "status": "FAIL",
                    "reason": "Invalid sampled row width",
                    "sampled_rows_checked": sampled_count,
                    "matched_rows": matched_count,
                }

            found = False

            for original_row in original_iterator:

                original_count += 1

                if len(original_row) != len(EXPECTED_HEADER):
                    continue

                if original_row == sampled_row:
                    found = True
                    matched_count += 1
                    break

            if not found:
                return {
                    "status": "FAIL",
                    "reason": (
                        "Sampled record not found in original "
                        "at the required relative position"
                    ),
                    "sampled_rows_checked": sampled_count,
                    "matched_rows": matched_count,
                    "original_rows_scanned": original_count,
                }

    return {
        "status": "PASS",
        "reason": "All sampled rows matched in original order",
        "sampled_rows_checked": sampled_count,
        "matched_rows": matched_count,
        "original_rows_scanned": original_count,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(SUMMARY_FILE)

    import pandas as pd

    summary = pd.read_csv(SUMMARY_FILE)

    results = []

    print("=" * 65)
    print("SAMPLED ROW CORRESPONDENCE VERIFICATION")
    print("=" * 65)

    for _, item in summary.iterrows():

        partition = str(item["partition"])
        filename = str(item["file"])

        original_path = ORIGINAL_ROOT / partition / filename
        sampled_path = SAMPLED_ROOT / partition / filename

        print(f"\nChecking {partition}/{filename}")

        if not original_path.exists() or not sampled_path.exists():
            result = {
                "partition": partition,
                "file": filename,
                "status": "FAIL",
                "reason": "Original or sampled file missing",
            }
        else:
            verification = verify_file(
                original_path,
                sampled_path,
            )

            result = {
                "partition": partition,
                "file": filename,
                **verification,
            }

        results.append(result)

        print(f"  Status: {result['status']}")
        print(f"  Reason: {result['reason']}")

        if "matched_rows" in result:
            print(
                f"  Matched rows: "
                f"{result['matched_rows']:,}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "04_row_correspondence_results.csv",
        index=False,
    )

    passed = int((results_df["status"] == "PASS").sum())
    failed = len(results_df) - passed

    report = {
        "files_checked": len(results_df),
        "files_passed": passed,
        "files_failed": failed,
        "overall_status": "PASS" if failed == 0 else "FAIL",
        "verification_method": (
            "Streaming sequential record matching. "
            "Every sampled CSV record must match a record "
            "in its original CSV at or after the previous match."
        ),
        "limitations": [
            "Identical duplicate records may match an earlier "
            "indistinguishable occurrence.",
            "This verifies ordered subsequence correspondence, "
            "not the exact original row indices selected."
        ],
    }

    with open(
        OUTPUT_DIR / "05_row_correspondence_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("VERIFICATION COMPLETE")
    print("=" * 65)
    print(f"Files checked: {len(results_df)}")
    print(f"Passed:        {passed}")
    print(f"Failed:        {failed}")
    print(f"Overall:       {report['overall_status']}")
    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
