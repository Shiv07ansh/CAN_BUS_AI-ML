
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "prepared" / "set_01"

OUTPUT_DIR = (
    ROOT / "reports" / "set_01" / "stage_6"
    / "feature_relevance"
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
MAX_SAMPLE = 100_000
SEED = 42


# ============================================================
# LOAD A REPRODUCIBLE TRAINING SAMPLE
# ============================================================

def load_training_sample():

    rng = np.random.default_rng(SEED)
    sampled_chunks = []

    total_rows = 0

    for filename in TRAINING_FILES:

        matches = list(DATA_DIR.rglob(filename))

        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {filename}; found {len(matches)}"
            )

        filepath = matches[0]

        print(f"Reading {filename}")

        for chunk in pd.read_csv(
            filepath,
            chunksize=CHUNK_SIZE
        ):

            required = FEATURES + [TARGET]

            chunk = chunk[required].apply(
                pd.to_numeric,
                errors="raise"
            )

            if chunk.isna().any().any():
                raise ValueError(
                    f"Missing values found in {filename}"
                )

            if not chunk[TARGET].isin([0, 1]).all():
                raise ValueError(
                    f"Invalid labels found in {filename}"
                )

            n = min(SAMPLE_PER_CHUNK, len(chunk))

            sampled_chunks.append(
                chunk.sample(
                    n=n,
                    random_state=int(rng.integers(0, 2**31 - 1))
                )
            )

            total_rows += len(chunk)

    sample = pd.concat(
        sampled_chunks,
        ignore_index=True
    )

    if len(sample) > MAX_SAMPLE:
        sample = sample.sample(
            n=MAX_SAMPLE,
            random_state=SEED
        ).reset_index(drop=True)

    print(f"\nTraining records inspected: {total_rows:,}")
    print(f"Analysis sample: {len(sample):,}")

    return sample


# ============================================================
# FEATURE RELEVANCE
# ============================================================

def main():

    print("=" * 65)
    print("SUPERVISED FEATURE RELEVANCE ANALYSIS")
    print("=" * 65)

    data = load_training_sample()

    X = data[FEATURES].astype(float)
    y = data[TARGET].astype(int)

    class_counts = y.value_counts().to_dict()

    if len(class_counts) != 2:
        raise ValueError(
            "Both normal and attack classes are required."
        )

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    # --------------------------------------------------------
    # 1. ANOVA F-TEST
    # --------------------------------------------------------

    print("\nCalculating ANOVA F-scores...")

    f_scores, p_values = f_classif(X, y)

    # --------------------------------------------------------
    # 2. MUTUAL INFORMATION
    # --------------------------------------------------------

    print("Calculating mutual information...")

    # All ten inputs are integer-valued features.
    # Treat them as discrete for this MI estimate.
    mi_scores = mutual_info_classif(
        X,
        y,
        discrete_features=True,
        random_state=SEED
    )

    # --------------------------------------------------------
    # 3. CLASS-CONDITIONAL DESCRIPTIVES
    # --------------------------------------------------------

    normal_means = X[y == 0].mean()
    attack_means = X[y == 1].mean()

    normal_medians = X[y == 0].median()
    attack_medians = X[y == 1].median()

    results = pd.DataFrame({
        "feature": FEATURES,
        "anova_f_score": f_scores,
        "anova_p_value": p_values,
        "mutual_information": mi_scores,
        "normal_mean": normal_means.values,
        "attack_mean": attack_means.values,
        "normal_median": normal_medians.values,
        "attack_median": attack_medians.values,
    })

    results["anova_rank"] = (
        results["anova_f_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    results["mutual_information_rank"] = (
        results["mutual_information"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    results["rank_sum"] = (
        results["anova_rank"]
        + results["mutual_information_rank"]
    )

    results = results.sort_values(
        "rank_sum"
    )

    results.to_csv(
        OUTPUT_DIR / "01_feature_relevance_ranking.csv",
        index=False
    )

    # --------------------------------------------------------
    # 4. SAVE ANALYSIS METADATA
    # --------------------------------------------------------

    summary = {
        "training_files": TRAINING_FILES,
        "sample_size": int(len(data)),
        "normal_records_in_sample": int((y == 0).sum()),
        "attack_records_in_sample": int((y == 1).sum()),
        "attack_prevalence": float(y.mean()),
        "random_seed": SEED,
        "methods": [
            "ANOVA F-test",
            "Mutual information"
        ],
        "mutual_information_features_treated_as": "Discrete",
        "features_ranked": FEATURES,
        "feature_removal_performed": False,
        "interpretation": (
            "These rankings are preliminary relevance measures. "
            "They do not establish causal importance or justify "
            "automatic feature removal."
        )
    }

    with open(
        OUTPUT_DIR / "02_analysis_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    # --------------------------------------------------------
    # 5. CONSOLE RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FEATURE RELEVANCE RESULTS")
    print("=" * 65)

    print(
        results[
            [
                "feature",
                "anova_f_score",
                "anova_rank",
                "mutual_information",
                "mutual_information_rank",
                "rank_sum"
            ]
        ].to_string(index=False)
    )

    print("\nClass-conditional means:")
    print(
        results[
            ["feature", "normal_mean", "attack_mean"]
        ].to_string(index=False)
    )

    print("\nNo features have been removed.")

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
