# Temporal Analysis — Set 01

## Scope

- Original Set 01 CSV files
- Chunk-based processing
- Chunk size: 200,000
- Inter-arrival times calculated within each file
- Original row order preserved
- One-second message-count windows

## Records

- Total records: 44,659,557

## Timestamp ordering

- Negative inter-arrival intervals: 815
- Zero inter-arrival intervals: 549
- Positive inter-arrival intervals: 44,658,141

## Method

Inter-arrival time is calculated as the difference between the
timestamp of the current message and the preceding message in
the original file order.

Intervals are associated with the class label of the current
message. Only positive intervals are included in the sampled
inter-arrival quantile statistics.

The quantile calculations use a deterministic systematic sample:
one interval is retained for every 100 positive
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
