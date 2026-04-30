import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import ConcatDataset, Subset, DataLoader, random_split
from torchhd import functional, embeddings
from torchhd.datasets import EuropeanLanguages as Languages
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score

# --- Library Imports ---
sys.path.append('../') 
try:
    from conformalHDC.models import *
    from conformalHDC.methods import *
    from conformalHDC.utils import *
except ImportError:
    print("Warning: conformalHDC modules not found. Ensure '../' is in path.")

# Fixed Constants
EXP_NAME = "ood_languages"
REPETITIONS = 10
DIMENSIONS = 10_000
MAX_INPUT_SIZE = 128
BATCH_SIZE = 32
PADDING_IDX = 0

# ASCII mappings
ASCII_A = ord("a")
ASCII_Z = ord("z")
ASCII_SPACE = ord(" ")
NUM_TOKENS = (ASCII_Z - ASCII_A + 1) + 1 + 1 

# --- ID vs OOD Definition (Indo-European vs Non-Indo-European) ---
# ID: All Indo-European Branches (Germanic, Romance, Slavic, Baltic, Hellenic)
ID_LANG_STRINGS = [
    'English', 'German', 'Dutch', 'Swedish', 'Danish',
    'French', 'Italian', 'Spanish', 'Portuguese', 'Romanian',
    'Czech', 'Polish', 'Slovak', 'Slovenian', 'Bulgarian',
    'Latvian', 'Lithuanian',
    'Greek'
]

# OOD (Implicit): The Uralic Family (Non-Indo-European)
# 'Finnish', 'Estonian', 'Hungarian'

CONFORMAL_SCORES = ["sim", "discount"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################======== Utility Functions ========################

def char2int(char: str) -> int:
    a = ord(char)
    if a == ASCII_SPACE:
        return (ASCII_Z - ASCII_A + 1)
    if ASCII_A <= a <= ASCII_Z:
        return a - ASCII_A
    return (ASCII_Z - ASCII_A + 1) 

def transform(x: str) -> torch.Tensor:
    x = x.lower()
    x = re.sub(r"\s+", " ", x)
    x = x[:MAX_INPUT_SIZE]
    ids = [char2int(ch) + 1 for ch in x]
    if len(ids) < MAX_INPUT_SIZE:
        ids += [PADDING_IDX] * (MAX_INPUT_SIZE - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size, dim, padding_idx=0):
        super().__init__()
        self.symbol = embeddings.Random(vocab_size, dim, padding_idx=padding_idx)
        
    @torch.no_grad()
    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        x_ids = x_ids.to(self.symbol.weight.device)
        symbols = self.symbol(x_ids)          # [B, T, D]
        hv = functional.ngrams(symbols, n=3)  # [B, D]
        # Bipolarize sample hypervectors
        hv = functional.hard_quantize(hv)     
        return hv

@torch.no_grad()
def get_hvs_labels(loader, encoder):
    encoder.eval()
    all_hvs, all_lbls = [], []
    for samples, labels in loader:
        samples = samples.to(DEVICE)
        hvs = encoder(samples)
        all_hvs.append(hvs.cpu().numpy())
        all_lbls.append(labels.numpy())
    if len(all_hvs) == 0: 
        return np.array([]), np.array([])
    return np.concatenate(all_hvs), np.concatenate(all_lbls)

def build_prototypes_float(hvs, labels, id_classes, dim):
    """
    Builds prototypes using SUM + NORMALIZE (Float precision).
    Matches the nn.Module training behavior.
    """
    n_classes = len(id_classes)
    protos = np.zeros((n_classes, dim))
    
    # Map dataset label `lbl` to row `i` in the prototype matrix
    for i, lbl in enumerate(id_classes):
        idx = np.where(labels == lbl)[0]
        if len(idx) > 0:
            class_sum = np.sum(hvs[idx], axis=0)
            norm = np.linalg.norm(class_sum)
            if norm > 1e-8:
                protos[i] = class_sum / norm
            else:
                protos[i] = class_sum 
    return protos

def get_cosine_similarities(X, prototypes):
    """Computes standard cosine similarity between queries and prototypes [-1, 1]."""
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    P_norm = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-8)
    return X_norm @ P_norm.T


################======== Experiment Logic ========################

def run_single_experiment(random_state):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    
    # Data Loading
    data_root = "./data"
    train_ds_raw = Languages(data_root, train=True, transform=transform, download=True)
    test_ds_raw  = Languages(data_root, train=False, transform=transform, download=True)
    
    # Map Strings to Integer Labels
    all_class_names = train_ds_raw.classes
    name_to_idx = {name: i for i, name in enumerate(all_class_names)}
    
    # Verify we have all ID languages
    for name in ID_LANG_STRINGS:
        if name not in name_to_idx:
            raise ValueError(f"Language {name} not found in dataset: {all_class_names}")

    ID_LABELS = [name_to_idx[name] for name in ID_LANG_STRINGS]
    
    # Full dataset
    ds_full = ConcatDataset([train_ds_raw, test_ds_raw])
    
    all_targets = []
    for d in ds_full.datasets:
        all_targets.extend(d.targets)
    all_targets = np.array(all_targets)
    
    MAX_SAMPLES_PER_CLASS = 2000
    subsampled_indices = []

    rng = np.random.RandomState(random_state) # Using fixed RNG for consistency
    for cls_idx in range(len(all_class_names)):
        # Get all indices for this specific class
        cls_indices = np.where(all_targets == cls_idx)[0]
        rng.shuffle(cls_indices) 
        # Take up to the limit
        subsampled_indices.extend(cls_indices[:MAX_SAMPLES_PER_CLASS])
    
    subsample_mask = np.zeros(len(all_targets), dtype=bool)
    subsample_mask[subsampled_indices] = True

    # Filter ID vs OOD using the subsampled mask
    id_indices = np.where(np.isin(all_targets, ID_LABELS) & subsample_mask)[0]
    ood_indices = np.where(~np.isin(all_targets, ID_LABELS) & subsample_mask)[0]
    
    ds_id_full = Subset(ds_full, id_indices)
    ds_ood = Subset(ds_full, ood_indices)
    
    # Random Split on In-Distribution (ID): 80% Train, 15% Calib, 5% Test
    n_total = len(ds_id_full)
    n_train = int(0.8 * n_total)
    n_calib = int(0.15 * n_total)
    n_test  = n_total - n_train - n_calib
    
    ds_train, ds_calib, ds_test = random_split(
        ds_id_full, [n_train, n_calib, n_test],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    # Loaders
    ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    ld_calib = DataLoader(ds_calib, batch_size=BATCH_SIZE, shuffle=False)
    ld_test  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    ld_ood   = DataLoader(ds_ood, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Data loading and splitting complete.")
    sys.stdout.flush()
    
    # HDC Init & Encoding
    encoder = LanguageEncoder(NUM_TOKENS, DIMENSIONS, PADDING_IDX).to(DEVICE)
    
    train_hvs, train_y = get_hvs_labels(ld_train, encoder)
    cal_hvs, cal_y     = get_hvs_labels(ld_calib, encoder)
    test_hvs, test_y   = get_hvs_labels(ld_test, encoder)
    ood_hvs, ood_y     = get_hvs_labels(ld_ood, encoder)
    print("Encoding complete.")
    sys.stdout.flush()
    
    # Balance OOD size
    min_len = min(len(ood_hvs), len(test_hvs))
    ood_hvs, ood_y = ood_hvs[:min_len], ood_y[:min_len]

    # Build Prototypes
    protos_train = build_prototypes_float(train_hvs, train_y, ID_LABELS, DIMENSIONS)
    print("Prototypes built.")
    sys.stdout.flush()

    # Conformal Prediction
    chdc  = ConformalHDC(protos_train, ID_LABELS, sim_measure="cosine", random_state=random_state)
    
    exp_results = []
    
    # =========================================================
    #    Conformal HDC Methods (sim, discount)
    # =========================================================
    for stype in CONFORMAL_SCORES:
        chdc.compute_calib_scores(cal_hvs, cal_y, score_type=stype)
        
        p_vals_id = chdc.get_max_p_value(test_hvs, marginal=True)
        p_vals_ood = chdc.get_max_p_value(ood_hvs, marginal=True)
        
        y_true_roc = np.concatenate([np.ones(len(p_vals_id)), np.zeros(len(p_vals_ood))])
        y_scores_roc = np.concatenate([p_vals_id, p_vals_ood])
        
        ood_auroc = roc_auc_score(y_true_roc, y_scores_roc)
        
        exp_results.append({
            "method": f"chdc_{stype}",
            "random_state": random_state,
            "ood_auroc": ood_auroc
        })

    # =========================================================
    #    Non-Conformal Baselines (MaxSim & Energy Score)
    # =========================================================
    # Calculate cosine similarities to class prototypes
    sims_id = get_cosine_similarities(test_hvs, protos_train)
    sims_ood = get_cosine_similarities(ood_hvs, protos_train)
    
    y_true_roc = np.concatenate([np.ones(len(sims_id)), np.zeros(len(sims_ood))])

    # A) Maximum Similarity (MaxSim)
    maxsim_id = np.max(sims_id, axis=1)
    maxsim_ood = np.max(sims_ood, axis=1)
    
    y_scores_maxsim = np.concatenate([maxsim_id, maxsim_ood])
    auroc_maxsim = roc_auc_score(y_true_roc, y_scores_maxsim)

    exp_results.append({
        "method": "maxsim",
        "random_state": random_state,
        "ood_auroc": auroc_maxsim
    })
    
    # B) Energy Score (Temperature T=1)
    neg_energy_id = np.log(np.sum(np.exp(sims_id), axis=1))
    neg_energy_ood = np.log(np.sum(np.exp(sims_ood), axis=1))
    
    y_scores_energy = np.concatenate([neg_energy_id, neg_energy_ood])
    auroc_energy = roc_auc_score(y_true_roc, y_scores_energy)

    exp_results.append({
        "method": "energy",
        "random_state": random_state,
        "ood_auroc": auroc_energy
    })

    print("Finished OOD Evaluations.")
    sys.stdout.flush()
    
    return pd.DataFrame(exp_results)


# ---------------
# Main Execution
# ---------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ood_languages.py <seed_group_id>")
        sys.exit(1)

    seed_arg = int(sys.argv[1])

    out_dir = Path(f"./results/{EXP_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"seed{seed_arg}_ood.csv"

    print(f"Starting OOD job: Seed Group {seed_arg}, Reps {REPETITIONS}")
    sys.stdout.flush()
    
    results_list = []
    for i in tqdm(range(1, REPETITIONS + 1), desc="Repetitions"):
        print(f"\nRunning repetition {i}...")
        sys.stdout.flush()
        current_state = REPETITIONS * (seed_arg - 1) + i
        try:
            df_rep = run_single_experiment(current_state)
            results_list.append(df_rep)
        except Exception as e:
            print(f"Error state {current_state}: {e}")
            sys.stdout.flush()
            
    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)
        final_df.to_csv(outfile, index=False)
        print(f"\nOOD Results saved to {outfile}")
        sys.stdout.flush()
    else:
        print("No results generated.")
        sys.stdout.flush()
