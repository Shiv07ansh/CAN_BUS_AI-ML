
from pathlib import Path
from collections import Counter
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPARED_DIR = PROJECT_ROOT / "data" / "prepared" / "set_01"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "set_01"
    / "stage_6"
    / "feature_quality_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_FILES = [
    "accessory-2.csv",
    "attack-free-1.csv",
    "attack-free-2.csv",
    "DoS-2.csv",
    "force-neutral-2.csv",
    "rpm-1.csv",
    "standstill-1.csv",
]

FEATURES = [
    "arbitration_id",
    "payload_length",
    *[f"byte_{i}" for i in range(8)],
]

TARGET = "attack"

CHUNK_SIZE = 100_000
SAMPLE_PER_CHUNK = 5_000
MAX_ANALYSIS_SAMPLE = 100_000

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# FIND TRAINING FILES
# ============================================================

def find_training_files():

    found = {}

    for filename in TRAINING_FILES:

        matches = list(PREPARED_DIR.rglob(filename))

        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {filename}, "
                f"found {len(matches)}"
            )

        found[filename] = matches[0]

    return found


# ============================================================
# EXACT STREAMING STATISTICS
# ============================================================

def update_statistics(chunk, counters, totals):

    for feature in FEATURES:

        values = chunk[feature].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values found in {feature}"
            )

        totals[feature]["count"] += len(values)
        totals[feature]["sum"] += values.sum()
        totals[feature]["sum_sq"] += np.square(values).sum()

        totals[feature]["minimum"] = min(
            totals[feature]["minimum"],
            values.min()
        )

        totals[feature]["maximum"] = max(
            totals[feature]["maximum"],
            values.max()
        )

        counters[feature].update(values.tolist())


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 65)
    print("CAN FEATURE QUALITY AUDIT")
    print("=" * 65)

    files = find_training_files()

    counters = {
        feature: Counter()
        for feature in FEATURES
    }

    totals = {
        feature: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "minimum": float("inf"),
            "maximum": float("-inf"),
        }
        for feature in FEATURES
    }

    sampled_chunks = []

    total_rows = 0
    total_attacks = 0

    # --------------------------------------------------------
    # Read training recordings only
    # --------------------------------------------------------

    for filename, filepath in files.items():

        print(f"\nProcessing: {filename}")

        for chunk in pd.read_csv(
            filepath,
            chunksize=CHUNK_SIZE
        ):

            required = FEATURES + [TARGET]

            missing = set(required) - set(chunk.columns)

            if missing:
                raise ValueError(
                    f"{filename}: missing columns {missing}"
                )

            chunk[required] = chunk[required].apply(
                pd.to_numeric,
                errors="raise"
            )

            if chunk[required].isna().any().any():
                raise ValueError(
                    f"Missing values found in {filename}"
                )

            if not chunk[TARGET].isin([0, 1]).all():
                raise ValueError(
                    f"Invalid labels found in {filename}"
                )

            update_statistics(
                chunk,
                counters,
                totals
            )

            total_rows += len(chunk)
            total_attacks += int(chunk[TARGET].sum())

            # Proportional random sample from each chunk.
            sample_count = min(
                SAMPLE_PER_CHUNK,
                len(chunk)
            )

            sampled_chunks.append(
                chunk[required].sample(
                    n=sample_count,
                    random_state=RANDOM_SEED + total_rows
                )
            )

    # --------------------------------------------------------
    # Combine and cap analysis sample
    # --------------------------------------------------------

    sample = pd.concat(
        sampled_chunks,
        ignore_index=True
    )

    if len(sample) > MAX_ANALYSIS_SAMPLE:

        sample = sample.sample(
            n=MAX_ANALYSIS_SAMPLE,
            random_state=RANDOM_SEED
        )

    X = sample[FEATURES].astype(float)

    print("\nTraining records inspected:", f"{total_rows:,}")
    print("Attack records:", f"{total_attacks:,}")
    print("Analysis sample:", f"{len(sample):,}")

    # --------------------------------------------------------
    # 1. Variance and distribution audit
    # --------------------------------------------------------

    variance_rows = []

    for feature in FEATURES:

        stats = totals[feature]
        count = stats["count"]

        mean = stats["sum"] / count

        variance = (
            stats["sum_sq"] / count
            - mean ** 2
        )

        variance = max(variance, 0.0)

        value_counts = counters[feature]

        most_common = value_counts.most_common(2)

        top_value, top_count = most_common[0]

        second_count = (
            most_common[1][1]
            if len(most_common) > 1
            else 0
        )

        dominant_ratio = (
            top_count / second_count
            if second_count > 0
            else float("inf")
        )

        unique_count = len(value_counts)
        unique_percentage = unique_count / count * 100

        zero_variance = stats["minimum"] == stats["maximum"]

        near_zero_variance = (
            not zero_variance
            and dominant_ratio > 19
            and unique_percentage < 10
        )

        variance_rows.append({
            "feature": feature,
            "count": count,
            "unique_values": unique_count,
            "minimum": stats["minimum"],
            "maximum": stats["maximum"],
            "mean": mean,
            "variance": variance,
            "most_common_value": top_value,
            "most_common_percentage": top_count / count * 100,
            "second_most_common_percentage": second_count / count * 100,
            "dominant_value_ratio": dominant_ratio,
            "unique_percentage": unique_percentage,
            "zero_variance": zero_variance,
            "near_zero_variance": near_zero_variance,
        })

    variance_df = pd.DataFrame(variance_rows)

    variance_df.to_csv(
        OUTPUT_DIR / "01_feature_variance.csv",
        index=False
    )

    # --------------------------------------------------------
    # 2. Correlation analysis
    # --------------------------------------------------------

    pearson = X.corr(method="pearson")
    spearman = X.corr(method="spearman")

    pearson.to_csv(
        OUTPUT_DIR / "02_pearson_correlation.csv"
    )

    spearman.to_csv(
        OUTPUT_DIR / "03_spearman_correlation.csv"
    )

    correlation_rows = []

    for i, feature_a in enumerate(FEATURES):

        for feature_b in FEATURES[i + 1:]:

            p = pearson.loc[feature_a, feature_b]
            s = spearman.loc[feature_a, feature_b]

            correlation_rows.append({
                "feature_1": feature_a,
                "feature_2": feature_b,
                "pearson_correlation": p,
                "spearman_correlation": s,
                "absolute_pearson": abs(p),
                "absolute_spearman": abs(s),
                "high_pearson_correlation": abs(p) >= 0.80,
                "high_spearman_correlation": abs(s) >= 0.80,
            })

    correlation_pairs = pd.DataFrame(correlation_rows)

    correlation_pairs.sort_values(
        "absolute_pearson",
        ascending=False
    ).to_csv(
        OUTPUT_DIR / "04_correlation_pairs.csv",
        index=False
    )

    # --------------------------------------------------------
    # 3. VIF analysis
    # --------------------------------------------------------

    print("\nCalculating VIF...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    vif_rows = []

    for i, feature in enumerate(FEATURES):

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                vif = variance_inflation_factor(
                    X_scaled,
                    i
                )

        except Exception:
            vif = np.inf

        vif_rows.append({
            "feature": feature,
            "VIF": vif,
            "interpretation": (
                "Very high multicollinearity"
                if vif >= 10
                else "Potential multicollinearity"
                if vif >= 5
                else "Low multicollinearity"
            )
        })

    vif_df = pd.DataFrame(vif_rows)

    vif_df.to_csv(
        OUTPUT_DIR / "05_vif_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # 4. Audit summary
    # --------------------------------------------------------

    summary = {
        "training_files": list(files.keys()),
        "training_records_inspected": int(total_rows),
        "training_attack_records": int(total_attacks),
        "training_attack_prevalence": (
            total_attacks / total_rows
        ),
        "analysis_sample_size": int(len(sample)),
        "features": FEATURES,
        "zero_variance_features": variance_df.loc[
            variance_df["zero_variance"],
            "feature"
        ].tolist(),
        "near_zero_variance_features": variance_df.loc[
            variance_df["near_zero_variance"],
            "feature"
        ].tolist(),
        "high_vif_features": vif_df.loc[
            vif_df["VIF"] >= 5,
            "feature"
        ].tolist(),
        "random_seed": RANDOM_SEED,
        "note": (
            "No features were automatically removed. "
            "VIF and correlations are diagnostic evidence, "
            "not automatic deletion criteria."
        )
    }

    with open(
        OUTPUT_DIR / "06_audit_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)

    print("\nFeature variance:")
    print(
        variance_df[
            [
                "feature",
                "variance",
                "unique_values",
                "zero_variance",
                "near_zero_variance"
            ]
        ].to_string(index=False)
    )

    print("\nVIF:")
    print(
        vif_df.to_string(index=False)
    )

    print("\nHighly correlated feature pairs:")

    high_corr = correlation_pairs[
        correlation_pairs["high_pearson_correlation"]
        | correlation_pairs["high_spearman_correlation"]
    ]

    if high_corr.empty:
        print("None detected at |correlation| >= 0.80")
    else:
        print(high_corr.to_string(index=False))

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
