
from pathlib import Path
import json
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "sampled_stratified"
    / "set_01"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "prepared"
    / "set_01"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_3"
    / "feature_extraction"
)

CHUNK_SIZE = 200_000

INPUT_COLUMNS = [
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
# FEATURE TRANSFORMATIONS
# =========================================================

def parse_arbitration_id(value):
    """Convert a hexadecimal CAN identifier to an integer."""

    value = str(value).strip().lower()

    if value.startswith("0x"):
        value = value[2:]

    return int(value, 16)


def extract_payload_features(payload):
    """
    Extract payload length and eight byte values.

    Missing byte positions are represented by -1.
    Empty payloads have length 0 and all byte values -1.
    """

    payload = str(payload).strip()

    if len(payload) % 2 != 0:
        raise ValueError(
            f"Odd-length hexadecimal payload: {payload}"
        )

    if payload:
        raw_bytes = bytes.fromhex(payload)
    else:
        raw_bytes = b""

    if len(raw_bytes) > 8:
        raise ValueError(
            f"Payload exceeds 8 bytes: {payload}"
        )

    features = {
        "payload_length": len(raw_bytes)
    }

    for i in range(8):
        if i < len(raw_bytes):
            features[f"byte_{i}"] = raw_bytes[i]
        else:
            features[f"byte_{i}"] = -1

    return features


def transform_chunk(chunk):

    if list(chunk.columns) != INPUT_COLUMNS:
        raise ValueError(
            f"Unexpected input schema: {list(chunk.columns)}"
        )

    output = pd.DataFrame(index=chunk.index)

    # Arbitration ID
    output["arbitration_id"] = (
        chunk["arbitration_id"]
        .map(parse_arbitration_id)
        .astype("int64")
    )

    # Payload features
    payload_features = (
        chunk["data_field"]
        .map(extract_payload_features)
    )

    payload_df = pd.DataFrame(
        payload_features.tolist(),
        index=chunk.index,
    )

    output = pd.concat(
        [output, payload_df],
        axis=1,
    )

    # Target label
    labels = chunk["attack"].astype(str).str.strip()

    if not labels.isin(["0", "1"]).all():
        raise ValueError("Invalid attack label encountered.")

    output["attack"] = labels.astype("int8")

    return output[OUTPUT_COLUMNS]


# =========================================================
# PROCESS ONE FILE
# =========================================================

def process_file(input_path, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    normal_count = 0
    attack_count = 0
    payload_length_counts = {}

    first_chunk = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        transformed = transform_chunk(chunk)

        transformed.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False

        total_rows += len(transformed)

        normal_count += int(
            (transformed["attack"] == 0).sum()
        )

        attack_count += int(
            (transformed["attack"] == 1).sum()
        )

        lengths = transformed["payload_length"].value_counts()

        for length, count in lengths.items():
            payload_length_counts[int(length)] = (
                payload_length_counts.get(int(length), 0)
                + int(count)
            )

        print(
            f"    Chunk {chunk_number}: "
            f"{len(transformed):,} rows processed"
        )

    return {
        "file": input_path.name,
        "rows": total_rows,
        "normal": normal_count,
        "attack": attack_count,
        "payload_length_counts": json.dumps(
            payload_length_counts,
            sort_keys=True,
        ),
        "status": "PASS",
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_ROOT}"
        )

    files = sorted(INPUT_ROOT.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_ROOT}"
        )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("=" * 60)
    print("CAN BUS FEATURE EXTRACTION")
    print("=" * 60)

    print(f"Input files: {len(files)}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print(f"Output: {OUTPUT_ROOT}")

    for index, input_path in enumerate(files, start=1):

        relative_path = input_path.relative_to(INPUT_ROOT)
        output_path = OUTPUT_ROOT / relative_path

        print(
            f"\n[{index}/{len(files)}] "
            f"{relative_path}"
        )

        result = process_file(
            input_path,
            output_path,
        )

        result["partition"] = relative_path.parts[0]

        results.append(result)

        print(
            f"  Completed: {result['rows']:,} rows | "
            f"Normal: {result['normal']:,} | "
            f"Attack: {result['attack']:,}"
        )

    # Save processing report
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_ROOT / "01_feature_extraction_summary.csv",
        index=False,
    )

    total_rows = int(results_df["rows"].sum())
    total_normal = int(results_df["normal"].sum())
    total_attack = int(results_df["attack"].sum())

    summary = {
        "files_processed": len(results_df),
        "total_rows": total_rows,
        "normal_records": total_normal,
        "attack_records": total_attack,
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": "attack",
        "output_columns": OUTPUT_COLUMNS,
        "overall_status": (
            "PASS"
            if total_rows > 0
            and total_normal + total_attack == total_rows
            else "FAIL"
        ),
    }

    with open(
        REPORT_ROOT / "02_feature_extraction_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Files processed: {len(results_df)}")
    print(f"Total rows:      {total_rows:,}")
    print(f"Normal:          {total_normal:,}")
    print(f"Attack:          {total_attack:,}")
    print(f"Status:          {summary['overall_status']}")
    print(f"\nPrepared data: {OUTPUT_ROOT}")
    print(f"Reports:       {REPORT_ROOT}")


if __name__ == "__main__":
    main()
