import numpy as np
import pandas as pd 
import pickle 
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split
from .config import DATA_ROOT


################======== UCI HAR ========################
def load_har_data():
    """Load UCI HAR, preserving train-then-test order and zero-based labels."""
    path = DATA_ROOT / "UCI_HAR"
    X_train = pd.read_csv(path / "train/X_train.txt", sep=r'\s+', header=None).values
    y_train = pd.read_csv(path / "train/y_train.txt", header=None).values.flatten() - 1
    X_test = pd.read_csv(path / "test/X_test.txt", sep=r'\s+', header=None).values
    y_test = pd.read_csv(path / "test/y_test.txt", header=None).values.flatten() - 1
    return np.vstack([X_train, X_test]), np.concatenate([y_train, y_test])


################======== Rat ========################
# --------------------
# Data Structures
# --------------------
@dataclass
class Split:
    """Holds the data for a single subset (e.g. Train)."""
    X: np.ndarray
    y: np.ndarray
    trial: np.ndarray
    idx: np.ndarray

@dataclass
class ThreeWaySplit:
    """
    Holds the three splits for the experiment.
    Nomenclature:
      - train: Model fitting
      - cal:   Conformal Calibration (Score computation)
      - test:  Final Evaluation
    """
    train: Split
    cal:   Split 
    test:  Split

# ---------------------------
# The Splitting Function
# ---------------------------
def split_three(X, y, trial, ratios=(0.45, 0.45, 0.1), stratify=False, *, seed=0):
    """
    Split (X, y) into 3 segments by ratios with optional stratification.
    
    Split rules:
        1. TRAIN vs REST: Split based on TRIALS (keeping all sub-windows per trial together).
           If stratify=True, we ensure the Train set has balanced classes of trials.
        2. CAL vs TEST: Split the REST data based on trials.
           If stratify=True, we ensure Cal and Test have balanced classes of samples.

    INPUTS:
        X: shape (n_samples, time_bins, n_features)
        y: shape (n_samples,) labels
        trial: shape (n_samples,) trial ID for each sample
        ratios: (train_ratio, cal_ratio, test_ratio)
        stratify: bool, whether to stratify the splits based on y labels.
    """

    # ---------------------
    #    Sanity Checks 
    # ---------------------
    tr, ca, te = ratios
    s = tr + ca + te 
    
    if any(r < 0 for r in (tr, ca, te)):
        raise ValueError("ratios must be non-negative")
    tr, ca, te = tr / s, ca / s, te / s

    n_samples = len(y)
    if not X.shape[0] == n_samples: 
        raise ValueError("X and y must have the same length")

    # -----------------------------------
    #    Split Trials (Train vs Rest)
    # -----------------------------------
    unique_trials, unique_indices = np.unique(trial, return_index=True)
    unique_y = y[unique_indices]

    stratify_labels = unique_y if stratify else None

    # Split the unique trials to make sure all subwindows per trial go to the same split
    train_trials, rest_trials = train_test_split(
        unique_trials, 
        test_size=(1 - tr), 
        random_state=seed, 
        stratify=stratify_labels
    )

    # --------------------------------
    #    Split Rest (Cal vs Test)
    # --------------------------------    
    cal_relative_ratio = ca / (ca + te)

    if stratify:
        # Get labels for the remaining trials to allow stratification
        mask_rest = np.isin(unique_trials, rest_trials)
        y_rest = unique_y[mask_rest]
    else:
        y_rest = None

    cal_trials, te_trials = train_test_split(
        rest_trials,
        train_size=cal_relative_ratio,
        random_state=seed,
        stratify=y_rest
    )

    # ---------------
    #   Packaging 
    # ---------------
    # Select ALL samples that belong to the chosen trials
    train_idx = np.where(np.isin(trial, train_trials))[0]
    cal_idx   = np.where(np.isin(trial, cal_trials))[0]
    te_idx    = np.where(np.isin(trial, te_trials))[0]

    def pack(idx):
        return Split(
            X = X[idx], 
            y = y[idx],
            trial = trial[idx],
            idx = idx
        )

    return ThreeWaySplit(
        train=pack(train_idx),
        cal=pack(cal_idx),  
        test=pack(te_idx),
    )
     
def summary(s: ThreeWaySplit):
    """Helper to print shapes of the 3 splits."""
    def fmt(split):
        return f"X: {split.X.shape}, {split.X.shape[0]} samples from {len(np.unique(split.trial))} trials."
    
    return {
        'Train': fmt(s.train),
        'Cal':   fmt(s.cal),
        'Test':  fmt(s.test)
    }


# -----------
# Loaders
# -----------
def prep_loader(irat, split_ratio, in_path, seed=123):
    """Legacy loader (no slicing)."""
    with in_path.open("rb") as f:
        odor_data = pickle.load(f)

    X = odor_data[irat]['binned_spk']
    y = odor_data[irat]['y'] 
    trial = odor_data[irat]['trial_id'] 
 
    splits = split_three(X, y, trial, ratios=split_ratio, seed=seed)

    return splits


def apply_sliding_window(X, y, trials, full_bins, window_bins, step_bins):
    """
    Augments data by sliding a window over the time dimension.
    Preserves trial_id to ensure safe splitting later.
    
    Args:
        X: spike data of shape (n_trials, n_time_bins, n_neurons)
        y:  odor labels of shape (n_trials,)
        trials: trial ids of shape (n_trials,)
        full_bins: Original number of bins (e.g., 16 for 400ms)
        window_bins: Target window size (e.g., 8 for 200ms)
        step_bins: Stride (e.g., 2)
        
    Returns:
        X_new, y_new, trials_new
    """
    n_trials, n_time, n_neurons = X.shape
    
    if n_time < window_bins:
        raise ValueError(f"Data length ({n_time}) is smaller than target window ({window_bins})")
        
    # Calculate starting indices
    # e.g., if total=16, win=8, step=2 -> starts at 0, 2, 4, ...
    start_indices = range(0, n_time - window_bins + 1, step_bins)
    
    X_list = []
    y_list = []
    trial_list = []
    
    for start in start_indices:
        end = start + window_bins
        
        # Slice the time dimension
        X_slice = X[:, start:end, :] 
        
        X_list.append(X_slice)
        y_list.append(y)          # Duplicate labels
        trial_list.append(trials) # Duplicate trial IDs (Crucial for splitting!)
        
    # Stack along the sample axis
    # New shape: (n_trials * n_slices, window_bins, n_neurons)
    X_new = np.concatenate(X_list, axis=0)
    y_new = np.concatenate(y_list, axis=0)
    trials_new = np.concatenate(trial_list, axis=0)
    
    return X_new, y_new, trials_new


def prep_loader_slicing(irat, split_ratio, in_path, 
                        step_bins=2,          # Stride (Overlap amount)
                        slicing_window=200,   # Slicing window size (matches running)
                        bin_size=25,
                        seed=0):
    
    # Load Raw Data
    with in_path.open("rb") as f:
        odor_data = pickle.load(f)

    X = odor_data[irat]['binned_spk'] # (n_trial, n_bin, n_neuron)
    y = odor_data[irat]['y']          # (n_trial, )
    trial = odor_data[irat]['trial_id'] # (n_trial, )
    
    # Augment (Sliding Window)
    full_bins = X.shape[1] 
    window_bins = slicing_window // bin_size # e.g. 200 // 25 = 8 bins
    
    print(f"Original Shape: {X.shape}")
    print(f"Applying sliding window of {slicing_window} ms ({window_bins} bins)")
    
    X_aug, y_aug, trial_aug = apply_sliding_window(
        X, y, trial, 
        full_bins=full_bins, 
        window_bins=window_bins, 
        step_bins=step_bins
    )
    
    print(f"Augmented Shape: {X_aug.shape}")

    # Since all slices from Trial 1 share the same ID, they all go to Train OR Test.
    splits = split_three(X_aug, y_aug, trial_aug, ratios=split_ratio, seed=seed)
    
    return splits


def prep_ood_rat(in_path_ood,rat_id):
    # Load OOD Data
    with in_path_ood.open("rb") as f:
        run_data = pickle.load(f)
    X_ood = run_data[rat_id]['binned_spk']
    return X_ood


def load_mnist_data():
    """Return concatenated MNIST train/test datasets with the original transform."""
    from torch.utils.data import ConcatDataset
    from torchvision import datasets, transforms

    root = str(DATA_ROOT)
    transform = transforms.ToTensor()
    return ConcatDataset([
        datasets.MNIST(root=root, train=True, transform=transform, download=True),
        datasets.MNIST(root=root, train=False, transform=transform, download=True),
    ])


def load_isolet_data():
    """Return float32 features and sorted, zero-based int64 class labels."""
    from sklearn.datasets import fetch_openml

    iso = fetch_openml('isolet', version=1, as_frame=False, parser='auto',
                       data_home=str(DATA_ROOT / "openml"))
    X = iso['data'].astype(np.float32)
    y = iso['target']
    classes = sorted(np.unique(y).tolist())
    label_to_id = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_to_id[s] for s in y], dtype=np.int64)
    return X, y_int


def load_languages_data(transform):
    """Return language train/test datasets using the experiment's transform."""
    from torchhd.datasets import EuropeanLanguages

    root = str(DATA_ROOT)
    return (
        EuropeanLanguages(root, train=True, transform=transform, download=True),
        EuropeanLanguages(root, train=False, transform=transform, download=True),
    )


def load_rat_data(irat, split_ratio, training_window, running_window,
                  bin_size=25, step_bins=2, slicing_window=200, seed=0):
    """Resolve rat filenames centrally and reuse the existing preparation."""
    root = DATA_ROOT / "rat"
    splits = prep_loader_slicing(
        irat, split_ratio,
        root / f"odor_prep_{training_window}_{bin_size}.pickle",
        step_bins=step_bins, slicing_window=slicing_window,
        bin_size=bin_size, seed=seed,
    )
    X_ood = prep_ood_rat(root / f"run_prep_{running_window}_{bin_size}.pickle", irat)
    return splits, X_ood
