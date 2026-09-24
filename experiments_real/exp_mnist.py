"""MNIST: digits 0-5 are in-distribution, 6-9 are OOD."""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.load import load_mnist_data
from hdc_encoder.image import encode_binary_images, make_position_hvs
from experiments_real.common import (
    DEVICE, balance_ood, evaluate, loader_orders, log, main, seed_everything,
)

# Fixed Constants
EXP_NAME = "mnist"
DIM = 10_000
BATCH_SIZE = 512
LABELS_ID = [0, 1, 2, 3, 4, 5]
LABELS_OOD = [6, 7, 8, 9]


################======== Data ========################
@lru_cache(maxsize=1)
def load_binary_images():
    """Every MNIST image as flattened pixels >= 0.5 after ToTensor's /255, plus labels."""
    datasets = load_mnist_data().datasets
    pixels = torch.cat([d.data for d in datasets]).flatten(1)
    labels = torch.cat([d.targets for d in datasets]).numpy()
    return pixels.float().div(255) >= 0.5, labels


def balanced_indices(targets, rng):
    """Subsample every class, in label order, down to the smallest class size."""
    class_indices = [np.flatnonzero(targets == c) for c in np.unique(targets)]
    min_count = min(len(idxs) for idxs in class_indices)
    return np.concatenate([rng.choice(idxs, min_count, replace=False) for idxs in class_indices])


def split_indices(random_state):
    """Image indices of the train/calib/test (80/15/5% of balanced ID) and OOD splits."""
    _, targets = load_binary_images()
    balanced = balanced_indices(targets, np.random.RandomState(random_state))
    id_idx = balanced[np.isin(targets[balanced], LABELS_ID)]
    ood_idx = balanced[np.isin(targets[balanced], LABELS_OOD)]

    n_total = len(id_idx)
    n_train, n_calib = int(0.8 * n_total), int(0.15 * n_total)
    splits = random_split(
        range(n_total), [n_train, n_calib, n_total - n_train - n_calib],
        generator=torch.Generator().manual_seed(random_state),
    )
    return [id_idx[split.indices] for split in splits] + [ood_idx]


def encode(indices, pos_hvs):
    """(HVs, labels) of the images at indices."""
    pixels, targets = load_binary_images()
    return encode_binary_images(pixels[indices], pos_hvs, BATCH_SIZE), targets[indices]


################======== Experiment Logic ========################

def run_single_experiment(random_state, alpha):
    seed_everything(random_state)
    train_idx, calib_idx, test_idx, ood_idx = split_indices(random_state)
    pos_hvs = make_position_hvs(28 * 28, DIM, DEVICE)

    # Split first before encode
    *_, train_order = loader_orders([
        (len(train_idx), True), (len(train_idx) + len(calib_idx), True),
        (len(calib_idx), False), (len(test_idx), False), (len(ood_idx), False),
        (len(train_idx), True),
    ], BATCH_SIZE)

    train = encode(train_idx[train_order], pos_hvs)
    cal = encode(calib_idx, pos_hvs)
    test = encode(test_idx, pos_hvs)
    ood_hvs, _ = encode(balance_ood(ood_idx, len(test_idx)), pos_hvs)
    log("Encoding complete.")
    return evaluate(train, cal, test, ood_hvs, LABELS_ID, "bipolar", random_state, alpha)


if __name__ == "__main__":
    main(run_single_experiment, EXP_NAME)
