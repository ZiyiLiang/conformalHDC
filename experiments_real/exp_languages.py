"""European languages: Indo-European languages are in-distribution; the Uralic
family (Finnish, Estonian, Hungarian) is OOD."""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.load import load_languages_data
from hdc_encoder.text import TrigramEncoder
from experiments_real.common import (
    DEVICE, balance_ood, evaluate, loader_orders, log, main, seed_everything,
)

# Fixed Constants
EXP_NAME = "languages"
DIMENSIONS = 10_000
MAX_SAMPLES_PER_CLASS = 2000 # reduced because the computation time
LOADER_BATCH_SIZE = 32    # batch size of the original shuffled train loader

# ID: All Indo-European Branches (Germanic, Romance, Slavic, Baltic, Hellenic)
ID_LANG_STRINGS = [
    # Germanic
    'English', 'German', 'Dutch', 'Swedish', 'Danish',
    # Romance
    'French', 'Italian', 'Spanish', 'Portuguese', 'Romanian',
    # Slavic
    'Czech', 'Polish', 'Slovak', 'Slovenian', 'Bulgarian',
    # Baltic
    'Latvian', 'Lithuanian',
    # Hellenic
    'Greek'
]


################======== Data ========################

@lru_cache(maxsize=1)
def load_texts():
    """Train-then-test strings, their labels and the class names."""
    train, test = load_languages_data(transform=None)
    targets = np.concatenate([train.targets.numpy(), test.targets.numpy()])
    return train.data + test.data, targets, train.classes


def id_labels(class_names):
    missing = [name for name in ID_LANG_STRINGS if name not in class_names]
    if missing:
        raise ValueError(f"Languages {missing} not found in dataset: {class_names}")
    return [class_names.index(name) for name in ID_LANG_STRINGS]


def split_indices(targets, labels_id, n_classes, random_state):
    """Subsample every class to MAX_SAMPLES_PER_CLASS, then split ID 75/22.5/2.5% into
    train/calib/test; the rest of the subsample is OOD."""
    keep = np.zeros(len(targets), dtype=bool)
    for c in range(n_classes):
        idx = np.flatnonzero(targets == c)
        np.random.shuffle(idx)  # global numpy RNG, seeded per repetition
        keep[idx[:MAX_SAMPLES_PER_CLASS]] = True
    is_id = np.isin(targets, labels_id)
    id_idx, ood_idx = np.flatnonzero(is_id & keep), np.flatnonzero(~is_id & keep)

    n_total = len(id_idx)
    n_train, n_calib = int(0.75 * n_total), int(0.225 * n_total)
    splits = random_split(
        range(n_total), [n_train, n_calib, n_total - n_train - n_calib],
        generator=torch.Generator().manual_seed(random_state),
    )
    return [id_idx[split.indices] for split in splits] + [ood_idx]


def encode(indices, encoder):
    """(HVs, labels) of the texts at indices."""
    texts, targets, _ = load_texts()
    return encoder.encode_texts([texts[i] for i in indices]), targets[indices]


################======== Experiment Logic ========################

def run_single_experiment(random_state, alpha):
    seed_everything(random_state)
    _, targets, class_names = load_texts()
    labels_id = id_labels(class_names)
    train_idx, calib_idx, test_idx, ood_idx = split_indices(targets, labels_id, len(class_names), random_state)
    log(f"ID: {len(train_idx)} train, {len(calib_idx)} calib, {len(test_idx)} test.")

    encoder = TrigramEncoder(DIMENSIONS).to(DEVICE)
    # The shuffled train loader was the first pass after creating the encoder;
    train_order, = loader_orders([(len(train_idx), True)], LOADER_BATCH_SIZE)
    train = encode(train_idx[train_order], encoder)
    cal = encode(calib_idx, encoder)
    test = encode(test_idx, encoder)
    ood_hvs, _ = encode(balance_ood(ood_idx, len(test_idx)), encoder)
    log("Encoding complete.")
    df = evaluate(train, cal, test, ood_hvs, labels_id, "normalized", random_state, alpha)
    # Take min label-conditional coverage of every set-valued row
    df.insert(df.columns.get_loc("lc_covs") + 1, "min_class_cov",
              df["lc_covs"].map(lambda v: min(v) if isinstance(v, list) else v))
    return df


if __name__ == "__main__":
    main(run_single_experiment, EXP_NAME)
