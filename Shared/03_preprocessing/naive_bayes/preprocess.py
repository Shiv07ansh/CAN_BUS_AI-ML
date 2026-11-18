
import sys
from pathlib import Path

PREPROCESSING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREPROCESSING_DIR))

from common.preprocessing_utils import preprocess_representation


def main():
    representations = ["E1", "E4", "E5", "B1"]

    for representation in representations:
        preprocess_representation(
            representation=representation,
            model_name="naive_bayes",
            scale_features=False
        )


if __name__ == "__main__":
    main()
