
from pathlib import Path
from itertools import combinations
import json
import pandas as pd

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_ROOT = ROOT / "data" / "prepared" / "set_01" / "train_01"

OUTPUT_ROOT = (
    ROOT / "reports" / "set_01"
    / "stage_4" / "file_level_split"
)

CHUNK_SIZE = 200_000
TARGET_VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

# Attack families identified from the training recording names.
ATTACK_FAMILIES = [
    "dos",
    "force-neutral",
    "rpm",
    "standstill",
]


# =========================================================
# FILE CLASSIFICATION
# =========================================================

def identify_family(filename):

    name = Path(filename).stem.lower()

    for family in ATTACK_FAMILIES:
        if name.startswith(family):
            return family

    return "normal_only"


# =========================================================
# COLLECT FILE STATISTICS
# =========================================================

def inspect_file(path):

    total = 0
    attacks = 0

    for chunk in pd.read_csv(
        path,
        usecols=["attack"],
        chunksize=CHUNK_SIZE,
    ):

        total += len(chunk)
        attacks += int(chunk["attack"].sum())

    return {
        "file": path.name,
        "relative_path": str(path.relative_to(TRAIN_ROOT)),
        "records": total,
        "normal": total - attacks,
        "attack": attacks,
        "attack_percentage": (
            100 * attacks / total if total else 0
        ),
        "family": identify_family(path.name),
    }


# =========================================================
# FIND BEST GROUP-LEVEL SPLIT
# =========================================================

def find_best_split(files_df):

    records = files_df["records"].sum()
    attacks = files_df["attack"].sum()

    overall_attack_rate = attacks / records

    best_split = None
    best_score = float("inf")

    file_count = len(files_df)

    # Enumerate all possible file-level assignments.
    # The dataset has only 12 files, so this is inexpensive.
    for mask in range(1, (1 << file_count) - 1):

        validation_indices = [
            i for i in range(file_count)
            if mask & (1 << i)
        ]

        validation = files_df.iloc[validation_indices]
        training = files_df.drop(validation.index)

        validation_records = int(validation["records"].sum())
        validation_attacks = int(validation["attack"].sum())
        validation_normal = int(validation["normal"].sum())

        training_records = int(training["records"].sum())
        training_attacks = int(training["attack"].sum())
        training_normal = int(training["normal"].sum())

        if not validation_records or not training_records:
            continue

        # Both groups must contain both classes.
        if validation_attacks == 0 or validation_normal == 0:
            continue

        if training_attacks == 0 or training_normal == 0:
            continue

                # Require attack-free recordings in both groups.
        if not (validation["family"] == "normal_only").any():
            continue

        if not (training["family"] == "normal_only").any():
            continue
        
        # Every attack family must appear in both groups.
        family_constraint_passed = True

        for family in ATTACK_FAMILIES:

            validation_family = validation[
                (validation["family"] == family)
                & (validation["attack"] > 0)
            ]

            training_family = training[
                (training["family"] == family)
                & (training["attack"] > 0)
            ]

            if validation_family.empty or training_family.empty:
                family_constraint_passed = False
                break

        if not family_constraint_passed:
            continue

        validation_fraction = validation_records / records

        # Avoid highly unbalanced holdouts.
        if not 0.10 <= validation_fraction <= 0.35:
            continue

        validation_attack_rate = (
            validation_attacks / validation_records
        )

        training_attack_rate = (
            training_attacks / training_records
        )

        # Primary objective: approach the 80/20 record split.
        size_error = abs(
            validation_fraction - TARGET_VALIDATION_FRACTION
        )

        # Secondary objective: keep class prevalence reasonably
        # close to the overall training partition.
        prevalence_error = (
            abs(validation_attack_rate - overall_attack_rate)
            / overall_attack_rate
        )

        # Also avoid a severe difference in attack prevalence
        # between the training and validation groups.
        group_prevalence_error = (
            abs(validation_attack_rate - training_attack_rate)
            / overall_attack_rate
        )

        score = (
            size_error
            + 0.05 * prevalence_error
            + 0.025 * group_prevalence_error
        )

        if score < best_score:
            best_score = score

            best_split = {
                "validation_indices": validation_indices,
                "score": score,
                "validation_fraction": validation_fraction,
                "validation_attack_rate": validation_attack_rate,
                "training_attack_rate": training_attack_rate,
            }

    if best_split is None:
        raise RuntimeError(
            "No feasible file-level split found. "
            "Inspect the attack-family file distribution."
        )

    return best_split


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted(TRAIN_ROOT.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No prepared training files found: {TRAIN_ROOT}"
        )

    print("=" * 65)
    print("FILE-LEVEL TRAINING / VALIDATION SPLIT")
    print("=" * 65)

    # Inspect files.
    file_statistics = []

    for index, path in enumerate(files, start=1):

        print(f"\nInspecting [{index}/{len(files)}]: {path.name}")

        stats = inspect_file(path)
        file_statistics.append(stats)

        print(
            f"  Rows: {stats['records']:,} | "
            f"Normal: {stats['normal']:,} | "
            f"Attack: {stats['attack']:,} | "
            f"Family: {stats['family']}"
        )

    files_df = pd.DataFrame(file_statistics)

    # Verify expected attack families.
    for family in ATTACK_FAMILIES:

        family_files = files_df[
            (files_df["family"] == family)
            & (files_df["attack"] > 0)
        ]

        if len(family_files) < 2:
            raise RuntimeError(
                f"At least two attack-containing files are "
                f"required for family: {family}"
            )

    # Find optimal feasible split.
    best = find_best_split(files_df)

    validation_indices = best["validation_indices"]

    files_df["group"] = "training"
    files_df.loc[validation_indices, "group"] = "validation"

    training_df = files_df[
        files_df["group"] == "training"
    ].copy()

    validation_df = files_df[
        files_df["group"] == "validation"
    ].copy()

    # Save file assignment.
    files_df.to_csv(
        OUTPUT_ROOT / "01_split_assignment.csv",
        index=False,
    )

    training_df.to_csv(
        OUTPUT_ROOT / "02_training_files.csv",
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_ROOT / "03_validation_files.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Group summaries
    # -----------------------------------------------------

    def group_summary(df):

        total = int(df["records"].sum())
        normal = int(df["normal"].sum())
        attack = int(df["attack"].sum())

        return {
            "files": len(df),
            "records": total,
            "normal": normal,
            "attack": attack,
            "attack_percentage": (
                100 * attack / total if total else 0
            ),
            "families_present": sorted(
                df.loc[df["attack"] > 0, "family"].unique().tolist()
            ),
        }

    training_summary = group_summary(training_df)
    validation_summary = group_summary(validation_df)

    # Verify that no files overlap.
    training_paths = set(training_df["relative_path"])
    validation_paths = set(validation_df["relative_path"])

    no_overlap = training_paths.isdisjoint(validation_paths)

    family_coverage = {}

    for family in ATTACK_FAMILIES:

        family_coverage[family] = {
            "training": int(
                (
                    (training_df["family"] == family)
                    & (training_df["attack"] > 0)
                ).sum()
            ),
            "validation": int(
                (
                    (validation_df["family"] == family)
                    & (validation_df["attack"] > 0)
                ).sum()
            ),
        }

    checks = {
        "no_file_overlap": no_overlap,
        "training_has_both_classes": (
            training_summary["normal"] > 0
            and training_summary["attack"] > 0
        ),
        "validation_has_both_classes": (
            validation_summary["normal"] > 0
            and validation_summary["attack"] > 0
        ),
        "all_attack_families_in_training": all(
            item["training"] > 0
            for item in family_coverage.values()
        ),
        "all_attack_families_in_validation": all(
            item["validation"] > 0
            for item in family_coverage.values()
        ),
    }

    overall_status = (
        "PASS" if all(checks.values()) else "FAIL"
    )

    summary = {
        "random_seed": RANDOM_SEED,
        "target_validation_fraction": TARGET_VALIDATION_FRACTION,
        "actual_validation_fraction": best["validation_fraction"],
        "selection_score": best["score"],
        "selection_method": (
            "Exhaustive file-level search over all possible "
            "assignments, subject to class and attack-family "
            "representation constraints."
        ),
        "training": training_summary,
        "validation": validation_summary,
        "attack_family_file_coverage": family_coverage,
        "checks": checks,
        "overall_status": overall_status,
        "important_note": (
            "Attack families are identified from source recording "
            "filenames for split design only. Filenames and family "
            "identifiers are not model input features. The split "
            "is a validation allocation within train_01; the four "
            "official test partitions remain untouched."
        ),
    }

    with open(
        OUTPUT_ROOT / "04_split_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4)

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("SELECTED SPLIT")
    print("=" * 65)

    print("\nTRAINING")
    print(f"Files:   {training_summary['files']}")
    print(f"Records: {training_summary['records']:,}")
    print(f"Normal:  {training_summary['normal']:,}")
    print(f"Attack:  {training_summary['attack']:,}")
    print(
        f"Attack %: {training_summary['attack_percentage']:.4f}%"
    )

    print("\nVALIDATION")
    print(f"Files:   {validation_summary['files']}")
    print(f"Records: {validation_summary['records']:,}")
    print(f"Normal:  {validation_summary['normal']:,}")
    print(f"Attack:  {validation_summary['attack']:,}")
    print(
        f"Attack %: {validation_summary['attack_percentage']:.4f}%"
    )

    print("\nATTACK FAMILY COVERAGE")

    for family, coverage in family_coverage.items():
        print(
            f"{family}: "
            f"training={coverage['training']} files, "
            f"validation={coverage['validation']} files"
        )

    print("\nVALIDATION CHECKS")

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print(f"\nOVERALL STATUS: {overall_status}")
    print(f"\nReports saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
