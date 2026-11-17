# Stage 1.3 — Class and Attack Distribution

## Overall distribution

- Files analysed: 52
- Total valid records: 44,659,557
- Normal records: 44,347,484
- Attack records: 312,073
- Attack prevalence: 0.698782%

## File representation

- Files containing attacks: 48
- Files containing no attacks: 4
- Files with highest attack prevalence: test_02_unknown_vehicle_known_attack\DoS-3.csv

## Partition distribution

| partition                              |   files |   records |   normal |   attack |   invalid_labels |   attack_percentage |   normal_percentage |   attack_to_normal_ratio |
|:---------------------------------------|--------:|----------:|---------:|---------:|-----------------:|--------------------:|--------------------:|-------------------------:|
| test_01_known_vehicle_known_attack     |       8 |   5702670 |  5639390 |    63280 |                0 |            1.10966  |             98.8903 |               0.0112211  |
| test_02_unknown_vehicle_known_attack   |       8 |   6447917 |  6283750 |   164167 |                0 |            2.54605  |             97.454  |               0.0261256  |
| test_03_known_vehicle_unknown_attack   |      12 |   8635275 |  8614450 |    20825 |                0 |            0.241162 |             99.7588 |               0.00241745 |
| test_04_unknown_vehicle_unknown_attack |      12 |  13220555 | 13206311 |    14244 |                0 |            0.107741 |             99.8923 |               0.00107858 |
| train_01                               |      12 |  10653140 | 10603583 |    49557 |                0 |            0.465187 |             99.5348 |               0.00467361 |

## Interpretation

The attack column provides binary labels only. Attack types cannot be distinguished from these labels alone.

Attack prevalence across source-row blocks is a descriptive diagnostic. It does not by itself establish attack episode boundaries or attack-generation methodology.

Original CSV files were not modified.