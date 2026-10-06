"""Fashion-MNIST: clothing (T-shirt, trouser, pullover, dress, coat, shirt) is
in-distribution; footwear and bags (sandal, sneaker, bag, ankle boot) are OOD."""
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.load import load_fashion_mnist_data
from experiments_real.common import main, prepare_binary_images, run_image_experiment

EXP_NAME = "fashion_mnist"
DIM = 10_000
BATCH_SIZE = 512
LABELS_ID = [0, 1, 2, 3, 4, 6]
LABELS_OOD = [5, 7, 8, 9]


@lru_cache(maxsize=1)
def load_binary_images():
    return prepare_binary_images(load_fashion_mnist_data().datasets)


def run_single_experiment(random_state, alpha):
    return run_image_experiment(
        *load_binary_images(), LABELS_ID, LABELS_OOD, random_state, alpha,
        dim=DIM, batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    main(run_single_experiment, EXP_NAME)
