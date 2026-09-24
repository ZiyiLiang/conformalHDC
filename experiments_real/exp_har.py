"""UCI HAR: dynamic activities (0-2) are in-distribution, static ones (3-5) are OOD."""
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.load import load_har_data
from hdc_encoder.level import quantize_to_levels
from experiments_real.common import main, run_level_experiment

EXP_NAME = "uci_har"
LABELS_ID = [0, 1, 2]
LABELS_OOD = [3, 4, 5]
LEVELS = 21
TEST_SIZE = 0.1

# TODO: runtime CSV (seed*_runtime.csv read by make_plots.R). Re-add timing of the
# encoding, prototype, calibration and prediction stages once the timing design is settled.


@lru_cache(maxsize=1)
def load_levels():
    X, y = load_har_data()
    return quantize_to_levels(X, LEVELS), y


def run_single_experiment(random_state, alpha):
    return run_level_experiment(
        *load_levels(), LABELS_ID, LABELS_OOD, TEST_SIZE, LEVELS, random_state, alpha,
    )


if __name__ == "__main__":
    main(run_single_experiment, EXP_NAME)
