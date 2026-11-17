# CAN Traffic Analysis — Set 01

## Scope

- Source: Original Set 01 CSV files
- Processing: Chunk-based, read-only
- Chunk size: 200,000
- Total records processed: 44,659,557

## Class distribution

- Normal records: 44,347,484
- Attack records: 312,073
- Attack prevalence: 0.698782%

## Payload validation

- Invalid payloads encountered during this analysis: 0
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
