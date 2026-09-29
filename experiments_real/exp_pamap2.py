"""
PAMAP2 intensity estimation

(Reiss & Stricker, 2012): the 12 protocol activities are grouped into light,
moderate and vigorous; light and vigorous are in-distribution and moderate is OOD.

"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.load import load_pamap2_data
from hdc_encoder.level import quantize_to_levels
from experiments_real.common import main, run_level_experiment

EXP_NAME = "pamap2"
# Activity indices of the protocol IDs [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]:
# lying, sitting, standing, walking, running, cycling, Nordic walking,
# ascending stairs, descending stairs, vacuum cleaning, ironing, rope jumping.
INTENSITY_ACTIVITIES = [
    [0, 1, 2, 10],       # light: lying, sitting, standing, ironing
    [3, 5, 6, 8, 9],     # moderate: walking, cycling, Nordic walking, descending stairs, vacuum cleaning
    [4, 7, 11],          # vigorous: running, ascending stairs, rope jumping
]
LABELS_ID = [0, 2]     # light, vigorous
LABELS_OOD = [1]       # moderate
LEVELS = 21
TEST_SIZE = 0.1

@lru_cache(maxsize=1)
def load_levels():
    X, activity = load_pamap2_data("Protocol")
    y = np.empty(len(activity), dtype=np.int64)
    for level, activities in enumerate(INTENSITY_ACTIVITIES):
        y[np.isin(activity, activities)] = level
    return quantize_to_levels(X, LEVELS), y


def run_single_experiment(random_state, alpha):
    return run_level_experiment(
        *load_levels(), LABELS_ID, LABELS_OOD, TEST_SIZE, LEVELS, random_state, alpha,
    )


if __name__ == "__main__":
    main(run_single_experiment, EXP_NAME)
