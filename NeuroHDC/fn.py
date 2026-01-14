import numpy as np  
import pickle 
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split

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

    X = odor_data[irat]['binned_spk'] # (n_trail, n_bin, n_neuron)
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




#=====================visual tools========================
import matplotlib.pyplot as plt
def plot_conf_mat(cm):
    '''visualize confusion matrix, row is actual, col is predicted'''

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()

    # annotate with 2 digits (integers if cm is int, else 2 decimals)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if np.issubdtype(cm.dtype, np.floating) else f"{cm[i, j]:02d}"
            plt.text(j, i, txt, ha="center", va="center")

    plt.xticks(range(cm.shape[1]))
    plt.yticks(range(cm.shape[0]))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.title('Averaged confusion matrix over 5 rats')
    plt.show()

def sim_matrix(H):
    # H complex (n,D), assumes unit magnitude-ish
    return (H @ np.conj(H).T).real / H.shape[1]

def plot_similarity_blocks(H, y, title="Cosine similarity matrix of class prototypes"):
    order = np.argsort(y)
    S = sim_matrix(H[order])
    plt.figure(figsize=(5, 4))
    plt.imshow(S, aspect="auto")
    plt.title(title)
    plt.colorbar() 
    plt.title(title)
 
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            plt.text(j, i,f'{S[i, j]:.2f}', ha="center", va="center")
    plt.xlabel('Odor class')
    plt.ylabel('Odor class')
    plt.show()


def check_dist(X,y):

    # Flatten time (treat the whole trial as one long vector for this check)
    X_flat = X.reshape(X.shape[0], -1) 

    # Calculate distances
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(X_flat)

    # Mask for Same vs Diff
    same_mask = y[:, None] == y[None, :]
    np.fill_diagonal(same_mask, False)
    diff_mask = ~same_mask

    avg_dist_same = dists[same_mask].mean()
    avg_dist_diff = dists[diff_mask].mean()

    print(f"Raw Euclidean Dist | Same: {avg_dist_same:.2f} | Diff: {avg_dist_diff:.2f}")

def check_beta_health(model, X, y, beta):
    # 1. Encode Data
    # Assuming W_neurons and Time_Base are already generated in model
    H = model.encode_all(model.W, X, model.TB, beta)
    
    # Compute similarity matrix (Real part of Hermitian product)
    # Sim(u, v) = Re(u . v*) / D
    gram = np.real(H @ H.conj().T) / model.D
    
    # 3. Mask for Same Class vs Diff Class
    # same_class_mask[i, j] is True if y[i] == y[j]
    same_class_mask = y[:, None] == y[None, :]
    np.fill_diagonal(same_class_mask, False) # Ignore self-similarity
    
    diff_class_mask = ~same_class_mask
    
    # 4. Compute Averages
    avg_intra = gram[same_class_mask].mean()
    avg_inter = gram[diff_class_mask].mean()
    
    print(f"Beta: {beta:.2f} | within : {avg_intra:.3f} | between : {avg_inter:.3f} | Delta: {avg_intra - avg_inter:.3f}")

# # Example Usage loop
# for b in [0.1, 0.15,0.2,0.25,0.3]:
#     check_beta_health(rff, Xtr_z, splits.train.y, b)