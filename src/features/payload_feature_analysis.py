
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.feature_selection import (
    f_classif,
    mutual_info_classif,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "engineered" / "set_01" / "train_01"

OUTPUT_DIR = (
    ROOT / "reports" / "set_01" / "stage_6"
    / "payload_feature_analysis"
)

PLOT_DIR = OUTPUT_DIR / "plots"

RANDOM_STATE = 42
ANALYSIS_SAMPLE_SIZE = 100_000

# Frozen training recordings from Stage 4.
TRAIN_FILES = [
    "accessory-2.csv",
    "attack-free-1.csv",
    "attack-free-2.csv",
    "DoS-2.csv",
    "force-neutral-2.csv",
    "rpm-1.csv",
    "standstill-1.csv",
]

ORIGINAL_FEATURES = [
    "arbitration_id",
    "payload_length",
    *[f"byte_{i}" for i in range(8)],
]

ENGINEERED_FEATURES = [
    "payload_zero_count",
    "payload_nonzero_count",
    "payload_byte_mean",
    "payload_byte_std",
    "payload_byte_entropy",
    "payload_unique_byte_count",
    "payload_bit_ones_count",
    "payload_bit_density",
]

FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES
TARGET = "attack"


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():
    frames = []

    print("Loading frozen training recordings...\n")

    for filename in TRAIN_FILES:
        path = INPUT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required training file not found: {path}"
            )

        df = pd.read_csv(path)

        required = FEATURES + [TARGET]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"{filename} is missing columns: {missing}"
            )

        df = df[required].copy()

        if df.isna().any().any():
            raise ValueError(
                f"Missing values detected in {filename}"
            )

        if not set(df[TARGET].unique()).issubset({0, 1}):
            raise ValueError(
                f"Unexpected target labels in {filename}"
            )

        df["source_file"] = filename
        frames.append(df)

        print(
            f"Loaded {filename}: {len(df):,} rows, "
            f"{int(df[TARGET].sum()):,} attacks"
        )

    data = pd.concat(frames, ignore_index=True)

    if not np.isfinite(
        data[FEATURES].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Non-finite feature values detected.")

    print("\nTraining data loaded successfully.")
    print(f"Total rows: {len(data):,}")
    print(f"Total attacks: {int(data[TARGET].sum()):,}")
    print(
        f"Attack prevalence: "
        f"{data[TARGET].mean() * 100:.4f}%"
    )

    return data


# ============================================================
# STRATIFIED ANALYSIS SAMPLE
# ============================================================

def create_analysis_sample(data):
    if len(data) <= ANALYSIS_SAMPLE_SIZE:
        sample = data.copy()
    else:
        sample, _ = train_test_split(
            data,
            train_size=ANALYSIS_SAMPLE_SIZE,
            stratify=data[TARGET],
            random_state=RANDOM_STATE,
        )

    sample = sample.reset_index(drop=True)

    print("\nAnalysis sample:")
    print(f"Rows: {len(sample):,}")
    print(f"Attacks: {int(sample[TARGET].sum()):,}")
    print(
        f"Attack prevalence: "
        f"{sample[TARGET].mean() * 100:.4f}%"
    )

    return sample


# ============================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(data):
    rows = []

    for feature in FEATURES:
        values = data[feature]

        rows.append({
            "feature": feature,
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
            "max": float(values.max()),
            "skewness": float(values.skew()),
        })

    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_DIR / "01_descriptive_statistics.csv",
        index=False,
    )

    # Class-wise descriptive statistics.
    class_rows = []

    for feature in FEATURES:
        for label, group in data.groupby(TARGET):
            values = group[feature]

            class_rows.append({
                "feature": feature,
                "class": int(label),
                "class_name": (
                    "attack" if label == 1 else "normal"
                ),
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
            })

    pd.DataFrame(class_rows).to_csv(
        OUTPUT_DIR / "02_classwise_descriptive_statistics.csv",
        index=False,
    )

    return result


# ============================================================
# 2. VARIANCE ANALYSIS
# ============================================================

def variance_analysis(data):
    rows = []

    for feature in FEATURES:
        values = data[feature]

        variance = float(values.var())
        unique_count = int(values.nunique())

        rows.append({
            "feature": feature,
            "variance": variance,
            "unique_values": unique_count,
            "zero_variance": unique_count <= 1,
            "near_zero_variance": (
                unique_count <= 2
                or variance < 1e-8
            ),
        })

    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_DIR / "03_variance_analysis.csv",
        index=False,
    )

    return result


# ============================================================
# 3. CLASS COMPARISON AND STATISTICAL SIGNIFICANCE
# ============================================================

def class_comparison(sample):
    normal = sample[sample[TARGET] == 0]
    attack = sample[sample[TARGET] == 1]

    rows = []

    for feature in FEATURES:
        normal_values = normal[feature].to_numpy()
        attack_values = attack[feature].to_numpy()

        test = mannwhitneyu(
            normal_values,
            attack_values,
            alternative="two-sided",
        )

        # Rank-biserial correlation derived from U.
        n_normal = len(normal_values)
        n_attack = len(attack_values)

        rank_biserial = (
            2 * test.statistic / (n_normal * n_attack)
        ) - 1

        rows.append({
            "feature": feature,
            "normal_median": float(np.median(normal_values)),
            "attack_median": float(np.median(attack_values)),
            "normal_mean": float(np.mean(normal_values)),
            "attack_mean": float(np.mean(attack_values)),
            "mann_whitney_u": float(test.statistic),
            "p_value": float(test.pvalue),
            "rank_biserial_correlation": float(rank_biserial),
            "absolute_effect_size": abs(float(rank_biserial)),
        })

    result = pd.DataFrame(rows)

    # Benjamini-Hochberg false discovery rate correction.
    p_values = result["p_value"].to_numpy()
    order = np.argsort(p_values)
    ranked = p_values[order]

    adjusted = np.empty(len(ranked), dtype=float)
    running_min = 1.0

    for i in range(len(ranked) - 1, -1, -1):
        candidate = (
            ranked[i] * len(ranked) / (i + 1)
        )
        running_min = min(running_min, candidate)
        adjusted[i] = min(running_min, 1.0)

    corrected = np.empty(len(p_values), dtype=float)
    corrected[order] = adjusted

    result["p_value_fdr"] = corrected
    result["significant_fdr_0_05"] = (
        result["p_value_fdr"] < 0.05
    )

    result.to_csv(
        OUTPUT_DIR / "04_class_comparison.csv",
        index=False,
    )

    return result


# ============================================================
# 4. CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(sample):
    x = sample[FEATURES]

    pearson = x.corr(method="pearson")
    spearman = x.corr(method="spearman")

    pearson.to_csv(
        OUTPUT_DIR / "05_pearson_correlation.csv"
    )

    spearman.to_csv(
        OUTPUT_DIR / "06_spearman_correlation.csv"
    )

    pairs = []

    for i in range(len(FEATURES)):
        for j in range(i + 1, len(FEATURES)):
            a = FEATURES[i]
            b = FEATURES[j]

            p = float(pearson.loc[a, b])
            s = float(spearman.loc[a, b])

            pairs.append({
                "feature_1": a,
                "feature_2": b,
                "pearson": p,
                "spearman": s,
                "absolute_pearson": abs(p),
                "absolute_spearman": abs(s),
            })

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values(
        "absolute_spearman",
        ascending=False,
    )

    pairs_df.to_csv(
        OUTPUT_DIR / "07_correlation_pairs.csv",
        index=False,
    )

    return pearson, spearman, pairs_df


# ============================================================
# 5. VARIANCE INFLATION FACTOR
# ============================================================

def calculate_vif(sample):
    """
    Calculate VIF using linear regression R-squared.

    This implementation avoids requiring statsmodels.
    """

    x = sample[FEATURES].to_numpy(dtype=np.float64)

    # Standardise for numerical stability.
    x = StandardScaler().fit_transform(x)

    rows = []

    for i, feature in enumerate(FEATURES):
        y = x[:, i]
        predictors = np.delete(x, i, axis=1)

        design = np.column_stack([
            np.ones(len(predictors)),
            predictors,
        ])

        coefficients, *_ = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )

        predictions = design @ coefficients

        residual_sum_squares = np.sum(
            (y - predictions) ** 2
        )

        total_sum_squares = np.sum(
            (y - y.mean()) ** 2
        )

        if total_sum_squares == 0:
            r_squared = 1.0
        else:
            r_squared = (
                1 - residual_sum_squares / total_sum_squares
            )

        r_squared = min(max(float(r_squared), 0.0), 1.0)

        if r_squared >= 0.999999999:
            vif = float("inf")
        else:
            vif = 1 / (1 - r_squared)

        rows.append({
            "feature": feature,
            "r_squared": r_squared,
            "vif": vif,
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_DIR / "08_vif_results.csv",
        index=False,
    )

    return result


# ============================================================
# 6. FEATURE RELEVANCE
# ============================================================

def feature_relevance(sample):
    x = sample[FEATURES]
    y = sample[TARGET]

    f_scores, f_p_values = f_classif(x, y)

    # Byte values and count features are discrete.
    discrete_features = [
        feature in {
            "arbitration_id",
            "payload_length",
            *[f"byte_{i}" for i in range(8)],
            "payload_zero_count",
            "payload_nonzero_count",
            "payload_unique_byte_count",
            "payload_bit_ones_count",
        }
        for feature in FEATURES
    ]

    mi_scores = mutual_info_classif(
        x,
        y,
        discrete_features=discrete_features,
        random_state=RANDOM_STATE,
    )

    result = pd.DataFrame({
        "feature": FEATURES,
        "anova_f_score": f_scores,
        "anova_p_value": f_p_values,
        "mutual_information": mi_scores,
    })

    result["anova_rank"] = (
        result["anova_f_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    result["mutual_information_rank"] = (
        result["mutual_information"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    result["combined_rank"] = (
        result["anova_rank"]
        + result["mutual_information_rank"]
    )

    result = result.sort_values("combined_rank")

    result.to_csv(
        OUTPUT_DIR / "09_feature_relevance.csv",
        index=False,
    )

    return result


# ============================================================
# 7. VISUALISATIONS
# ============================================================

def create_plots(data, sample, pearson, relevance):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Distribution plots
    # --------------------------------------------------------

    fig, axes = plt.subplots(6, 3, figsize=(17, 24))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURES):
        ax = axes[i]

        normal = sample.loc[
            sample[TARGET] == 0, feature
        ]

        attack = sample.loc[
            sample[TARGET] == 1, feature
        ]

        # Shared bins allow visual comparison.
        combined_min = min(normal.min(), attack.min())
        combined_max = max(normal.max(), attack.max())

        if combined_min == combined_max:
            bins = 10
        else:
            bins = np.linspace(
                combined_min,
                combined_max,
                31,
            )

        ax.hist(
            normal,
            bins=bins,
            density=True,
            alpha=0.55,
            label="Normal",
        )

        ax.hist(
            attack,
            bins=bins,
            density=True,
            alpha=0.55,
            label="Attack",
        )

        ax.set_title(feature)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for i in range(len(FEATURES), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Feature Distributions: Normal vs Attack",
        fontsize=16,
    )

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "01_feature_distributions.png",
        dpi=180,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Standardised class comparison boxplots
    # --------------------------------------------------------

    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample[FEATURES])

    scaled_df = pd.DataFrame(
        scaled,
        columns=FEATURES,
    )

    scaled_df[TARGET] = sample[TARGET].to_numpy()

    fig, ax = plt.subplots(figsize=(17, 8))

    normal_values = [
        scaled_df.loc[scaled_df[TARGET] == 0, f].to_numpy()
        for f in FEATURES
    ]

    attack_values = [
        scaled_df.loc[scaled_df[TARGET] == 1, f].to_numpy()
        for f in FEATURES
    ]

    positions_normal = np.arange(len(FEATURES)) * 3
    positions_attack = positions_normal + 1

    ax.boxplot(
        normal_values,
        positions=positions_normal,
        widths=0.65,
        showfliers=False,
    )

    ax.boxplot(
        attack_values,
        positions=positions_attack,
        widths=0.65,
        showfliers=False,
    )

    ax.set_xticks(positions_normal + 0.5)
    ax.set_xticklabels(
        FEATURES,
        rotation=55,
        ha="right",
    )

    ax.set_title(
        "Standardised Feature Values by Class"
    )
    ax.set_ylabel("Standardised value")
    ax.legend(
        handles=[],
        labels=[],
    )

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "02_class_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Correlation heatmap
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(
        pearson.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))

    ax.set_xticklabels(
        FEATURES,
        rotation=65,
        ha="right",
    )

    ax.set_yticklabels(FEATURES)

    ax.set_title("Pearson Correlation Heatmap")

    fig.colorbar(
        image,
        ax=ax,
        label="Pearson correlation",
    )

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "03_correlation_heatmap.png",
        dpi=180,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Feature relevance
    # --------------------------------------------------------

    plot_data = relevance.sort_values(
        "mutual_information",
        ascending=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    axes[0].barh(
        plot_data["feature"],
        plot_data["mutual_information"],
    )
    axes[0].set_title("Mutual Information")
    axes[0].set_xlabel("MI score")

    plot_data = relevance.sort_values(
        "anova_f_score",
        ascending=True,
    )

    axes[1].barh(
        plot_data["feature"],
        plot_data["anova_f_score"],
    )
    axes[1].set_title("ANOVA F-score")
    axes[1].set_xlabel("F-score")

    fig.tight_layout()
    fig.savefig(
        PLOT_DIR / "04_feature_relevance.png",
        dpi=180,
    )
    plt.close(fig)


# ============================================================
# 8. SUMMARY
# ============================================================

def create_summary(
    data,
    variance,
    comparison,
    pairs,
    vif,
    relevance,
):
    high_correlation = pairs[
        (pairs["absolute_pearson"] >= 0.90)
        | (pairs["absolute_spearman"] >= 0.90)
    ]

    summary = {
        "training_rows": int(len(data)),
        "training_attack_rows": int(data[TARGET].sum()),
        "training_normal_rows": int(
            (data[TARGET] == 0).sum()
        ),
        "attack_prevalence": float(data[TARGET].mean()),
        "number_of_features": len(FEATURES),
        "original_features": ORIGINAL_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "analysis_sample_size": ANALYSIS_SAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "zero_variance_features": variance.loc[
            variance["zero_variance"], "feature"
        ].tolist(),
        "near_zero_variance_features": variance.loc[
            variance["near_zero_variance"], "feature"
        ].tolist(),
        "features_significant_after_fdr": comparison.loc[
            comparison["significant_fdr_0_05"], "feature"
        ].tolist(),
        "highly_correlated_pairs": high_correlation[
            [
                "feature_1",
                "feature_2",
                "pearson",
                "spearman",
            ]
        ].to_dict(orient="records"),
        "vif_above_5": vif.loc[
            vif["vif"] > 5, "feature"
        ].tolist(),
        "top_features_by_combined_rank": relevance.head(
            10
        )["feature"].tolist(),
        "validation_data_used": False,
        "feature_removal_performed": False,
    }

    with open(
        OUTPUT_DIR / "10_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PAYLOAD FEATURE ANALYSIS")
    print("=" * 65)

    data = load_training_data()
    sample = create_analysis_sample(data)

    print("\n[1/8] Descriptive statistics...")
    descriptive_statistics(data)

    print("[2/8] Variance analysis...")
    variance = variance_analysis(data)

    print("[3/8] Class comparison and Mann-Whitney tests...")
    comparison = class_comparison(sample)

    print("[4/8] Pearson and Spearman correlations...")
    pearson, spearman, pairs = correlation_analysis(sample)

    print("[5/8] Variance inflation factors...")
    vif = calculate_vif(sample)

    print("[6/8] ANOVA and mutual information...")
    relevance = feature_relevance(sample)

    print("[7/8] Generating plots...")
    create_plots(data, sample, pearson, relevance)

    print("[8/8] Generating summary...")
    summary = create_summary(
        data,
        variance,
        comparison,
        pairs,
        vif,
        relevance,
    )

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)

    print(f"Training rows: {len(data):,}")
    print(f"Features analysed: {len(FEATURES)}")
    print(
        "Zero-variance features:",
        summary["zero_variance_features"],
    )
    print(
        "High-correlation pairs:",
        len(summary["highly_correlated_pairs"]),
    )
    print(
        "Features with VIF > 5:",
        summary["vif_above_5"],
    )
    print(
        "Top-ranked features:",
        summary["top_features_by_combined_rank"][:5],
    )

    print(f"\nReports: {OUTPUT_DIR}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
