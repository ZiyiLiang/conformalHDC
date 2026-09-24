"""ISOLET: letters 0-22 are in-distribution, 23-25 are OOD."""
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.load import load_isolet_data
from hdc_encoder.level import quantize_to_levels
from experiments_real.common import main, run_level_experiment

EXP_NAME = "isolet"
ID_CLASSES = list(range(23))
OOD_CLASSES = list(range(23, 26))
LEVELS = 21
TEST_SIZE = 0.05


@lru_cache(maxsize=1)
def load_levels():
    X, y = load_isolet_data()
    return quantize_to_levels(X, LEVELS), y


def run_single_experiment(random_state, alpha):
    return run_level_experiment(
        *load_levels(), ID_CLASSES, OOD_CLASSES, TEST_SIZE, LEVELS, random_state, alpha,
    )


if __name__ == "__main__":
    main(run_single_experiment, EXP_NAME)
