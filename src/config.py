"""Central paths and reproducibility settings for the project.

Team-written configuration: project paths and the seed utility.
"""

import random
from pathlib import Path

import numpy as np


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

HISTORICAL_FIGURE = "Marcus Aurelius"
DATASET_NAME = "Meditations"
RAW_TEXT_PATH = RAW_DATA_DIR / "marcus_aurelius_meditations.txt"
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
TEST_SET_PATH = PROCESSED_DATA_DIR / "team_test_set.json"
BASELINE_RESULTS_PATH = FIGURES_DIR / "baseline_results.csv"


def set_seed(seed: int = SEED) -> None:
    """Set Python and NumPy random seeds.

    Team-written utility. Scikit-learn receives this seed explicitly in
    deterministic operations that support a random_state parameter.
    """

    random.seed(seed)
    np.random.seed(seed)
