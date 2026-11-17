from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATHS = [
    ROOT / "models/set_01/baseline/random_forest.joblib",
    ROOT / "models/set_01/tuned/random_forest_tuned.joblib",
]

for path in MODEL_PATHS:

    print("\n" + "=" * 65)
    print(path.name)
    print("=" * 65)

    obj = joblib.load(path)

    print("Dictionary keys:", list(obj.keys()))

    for key, value in obj.items():

        print(f"\nKey: {key}")
        print("Type:", type(value))
        print("Class:", value.__class__.__name__)

        if hasattr(value, "feature_importances_"):
            print("FOUND RANDOM FOREST ESTIMATOR")
            print("Feature importances:", value.feature_importances_)

        if hasattr(value, "predict_proba"):
            print("Supports predict_proba")

        if hasattr(value, "named_steps"):
            print("Pipeline steps:", list(value.named_steps.keys()))

        if isinstance(value, dict):
            print("Nested dictionary keys:", list(value.keys()))