# Original set_01 — Data Quality Audit

## Scope

- Files inspected: 52
- Records inspected: 44,659,557
- Chunk size: 200,000
- Source CSV files were read-only.

## Findings

- Blank timestamps: 0
- Blank arbitration IDs: 0
- Blank payload fields: 915
- Blank labels: 0
- Invalid timestamps: 0
- Non-finite timestamps: 0
- Timestamp decreases: 815
- Invalid arbitration IDs: 0
- Out-of-range CAN IDs: 0
- Invalid payloads: 0
- Empty payloads: 915
- Odd-length payloads: 0
- Oversized payloads: 0
- Invalid labels: 0
- Within-chunk duplicate rows: 0
- Adjacent chunk-boundary duplicates: 0
- Conflicting label groups: 0

## Duplicate detection limitations

Exact duplicates are checked within individual chunks and at immediately adjacent chunk boundaries.
This does not detect all possible duplicates across distant chunks or separate files.

## Interpretation cautions

- Empty payloads are counted separately, not automatically classified as invalid.
- CAN IDs are checked against the 29-bit maximum; this does not establish whether a frame is standard or extended.
- Timestamp decreases are not automatically treated as corruption.
- Repeated CAN messages are not necessarily duplicate records.
- No source records were modified.

## Generated reports

- 01_file_quality_summary.csv
- 02_payload_length_distribution.csv
- 03_arbitration_id_frequency.csv
- 04_file_level_label_distribution.csv
- 05_payload_errors.csv
- 06_partition_quality_summary.csv