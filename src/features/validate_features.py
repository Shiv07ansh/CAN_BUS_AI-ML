
from pathlib import Path
import json
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = ROOT / "data" / "sampled_stratified" / "set_01"
PREPARED_ROOT = ROOT / "data" / "prepared" / "set_01"

REPORT_ROOT = (
    ROOT / "reports" / "set_01" / "stage_3"
    / "feature_validation"
)

CHUNK_SIZE = 200_000

SOURCE_COLUMNS = [
    "timestamp",
    "arbitration_id",
    "data_field",
    "attack",
]

FEATURE_COLUMNS = (
    ["arbitration_id", "payload_length"]
    + [f"byte_{i}" for i in range(8)]
)

OUTPUT_COLUMNS = FEATURE_COLUMNS + ["attack"]


# =========================================================
# EXPECTED FEATURE TRANSFORMATION
# =========================================================

def expected_features(source):

    ids = []
    lengths = []
    byte_columns = [[] for _ in range(8)]
    labels = []

    for row in source.itertuples(index=False, name=None):

        _, arbitration_id, payload, attack = row

        arbitration_id = str(arbitration_id).strip().lower()

        if arbitration_id.startswith("0x"):
            arbitration_id = arbitration_id[2:]

        ids.append(int(arbitration_id, 16))

        payload = str(payload).strip()

        raw_bytes = bytes.fromhex(payload) if payload else b""

        if len(raw_bytes) > 8:
            raise ValueError("Payload exceeds 8 bytes")

        lengths.append(len(raw_bytes))

        for i in range(8):
            byte_columns[i].append(
                raw_bytes[i] if i < len(raw_bytes) else -1
            )

        labels.append(int(attack))

    expected = pd.DataFrame({
        "arbitration_id": ids,
        "payload_length": lengths,
    })

    for i in range(8):
        expected[f"byte_{i}"] = byte_columns[i]

    expected["attack"] = labels

    return expected[OUTPUT_COLUMNS]


# =========================================================
# VALIDATE ONE FILE
# =========================================================

def validate_file(source_path, prepared_path):

    source_rows = 0
    prepared_rows = 0
    normal_count = 0
    attack_count = 0

    checks = {
        "schema": True,
        "numeric": True,
        "binary_labels": True,
        "payload_length_range": True,
        "byte_range": True,
        "sentinel_consistency": True,
        "no_missing_values": True,
        "exact_transformation": True,
    }

    errors = []

    source_reader = pd.read_csv(
        source_path,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    prepared_reader = pd.read_csv(
        prepared_path,
        chunksize=CHUNK_SIZE,
    )

    source_iterator = iter(source_reader)
    prepared_iterator = iter(prepared_reader)

    while True:

        source_chunk = next(source_iterator, None)
        prepared_chunk = next(prepared_iterator, None)

        if source_chunk is None and prepared_chunk is None:
            break

        if source_chunk is None or prepared_chunk is None:
            checks["exact_transformation"] = False
            errors.append("Source/prepared chunk count mismatch")
            break

        source_rows += len(source_chunk)
        prepared_rows += len(prepared_chunk)

        if list(source_chunk.columns) != SOURCE_COLUMNS:
            checks["schema"] = False

        if list(prepared_chunk.columns) != OUTPUT_COLUMNS:
            checks["schema"] = False
            errors.append("Prepared column order/schema mismatch")
            continue

        if len(source_chunk) != len(prepared_chunk):
            checks["exact_transformation"] = False
            errors.append("Chunk row count mismatch")
            continue

        if prepared_chunk.isna().any().any():
            checks["no_missing_values"] = False

        if not all(
            pd.api.types.is_numeric_dtype(prepared_chunk[col])
            for col in OUTPUT_COLUMNS
        ):
            checks["numeric"] = False

        if not prepared_chunk["attack"].isin([0, 1]).all():
            checks["binary_labels"] = False

        if not prepared_chunk["payload_length"].between(0, 8).all():
            checks["payload_length_range"] = False

        byte_values = prepared_chunk[
            [f"byte_{i}" for i in range(8)]
        ]

        if not byte_values.apply(
            lambda col: col.between(-1, 255).all()
        ).all():
            checks["byte_range"] = False

        lengths = prepared_chunk["payload_length"].to_numpy()
        byte_array = byte_values.to_numpy()

        for i in range(8):
            expected_missing = lengths <= i
            actual_missing = byte_array[:, i] == -1

            if not np.array_equal(expected_missing, actual_missing):
                checks["sentinel_consistency"] = False
                break

        normal_count += int(
            (prepared_chunk["attack"] == 0).sum()
        )

        attack_count += int(
            (prepared_chunk["attack"] == 1).sum()
        )

        # Independently reconstruct expected features
        expected = expected_features(source_chunk)

        actual = prepared_chunk[OUTPUT_COLUMNS].reset_index(drop=True)

        expected = expected.reset_index(drop=True)

        if not np.array_equal(
            actual.to_numpy(),
            expected.to_numpy(),
        ):
            checks["exact_transformation"] = False
            errors.append("Feature values differ from source transformation")

    status = "PASS" if all(checks.values()) else "FAIL"

    return {
        "status": status,
        "source_rows": source_rows,
        "prepared_rows": prepared_rows,
        "normal_records": normal_count,
        "attack_records": attack_count,
        **checks,
        "errors": "; ".join(errors),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    source_files = sorted(SOURCE_ROOT.rglob("*.csv"))

    if not source_files:
        raise FileNotFoundError("No sampled source CSVs found")

    results = []

    print("=" * 65)
    print("FEATURE MATRIX VALIDATION")
    print("=" * 65)

    for index, source_path in enumerate(source_files, start=1):

        relative = source_path.relative_to(SOURCE_ROOT)
        prepared_path = PREPARED_ROOT / relative

        print(f"\n[{index}/{len(source_files)}] {relative}")

        if not prepared_path.exists():

            result = {
                "status": "FAIL",
                "errors": "Prepared file missing",
            }

        else:
            result = validate_file(
                source_path,
                prepared_path,
            )

        result["partition"] = relative.parts[0]
        result["file"] = source_path.name

        results.append(result)

        print(f"  Status: {result['status']}")

        if result.get("errors"):
            print(f"  Errors: {result['errors']}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_ROOT / "01_feature_validation_results.csv",
        index=False,
    )

    passed = int((results_df["status"] == "PASS").sum())
    failed = len(results_df) - passed

    total_source = int(results_df["source_rows"].sum())
    total_prepared = int(results_df["prepared_rows"].sum())

    summary = {
        "files_checked": len(results_df),
        "files_passed": passed,
        "files_failed": failed,
        "source_records": total_source,
        "prepared_records": total_prepared,
        "normal_records": int(results_df["normal_records"].sum()),
        "attack_records": int(results_df["attack_records"].sum()),
        "overall_status": "PASS" if failed == 0 else "FAIL",
        "checks": list(checks := [
            "schema",
            "numeric",
            "binary_labels",
            "payload_length_range",
            "byte_range",
            "sentinel_consistency",
            "no_missing_values",
            "exact_transformation",
        ]),
    }

    with open(
        REPORT_ROOT / "02_feature_validation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETE")
    print("=" * 65)
    print(f"Files checked: {len(results_df)}")
    print(f"Passed:        {passed}")
    print(f"Failed:        {failed}")
    print(f"Source rows:   {total_source:,}")
    print(f"Prepared rows: {total_prepared:,}")
    print(f"Overall:       {summary['overall_status']}")


if __name__ == "__main__":
    main()
