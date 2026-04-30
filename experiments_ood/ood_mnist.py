import sys
import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision import datasets, transforms
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
EXP_NAME = "ood_mnist"
REPETITIONS = 10
DIM = 10_000
BATCH_SIZE = 512
LABELS_ID = [0, 1, 2, 3, 4, 5]
LABELS_OOD = [6, 7, 8, 9]

CONFORMAL_SCORES = ["sim", "discount"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################======== Utility Functions ========################
@torch.no_grad()
def bipolar_sign(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

def make_position_hvs(n_positions, dim, device=None):
    device = device or DEVICE
    return (torch.randint(0, 2, (n_positions, dim), device=device, dtype=torch.float32) * 2 - 1)

@torch.no_grad()
def encode_binary_images_to_hvs(imgs, pos_hvs, threshold=0.5):
    B, C, H, W = imgs.shape
    x = (imgs.view(B, -1) >= threshold).to(torch.float32).to(pos_hvs.device)
    return bipolar_sign(x @ pos_hvs)

@torch.no_grad()
def build_prototypes(loader, pos_hvs, class_labels):
    D = pos_hvs.shape[1]
    n_classes = len(class_labels)
    accum = torch.zeros((n_classes, D), device=pos_hvs.device)
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(pos_hvs.device), labels.to(pos_hvs.device)
        hvs = encode_binary_images_to_hvs(imgs, pos_hvs)
        for i, lbl in enumerate(class_labels):
            m = (labels == lbl)
            if m.any():
                accum[i] += hvs[m].sum(dim=0)
    return bipolar_sign(accum)

def get_hvs_labels(loader, pos_hvs):
    all_hvs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(pos_hvs.device)
        hvs = encode_binary_images_to_hvs(imgs, pos_hvs)
        all_hvs.append(hvs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_hvs), np.concatenate(all_labels)

# Baseline Helper
def get_cosine_similarities(X, prototypes):
    """Computes standard cosine similarity between queries and prototypes [-1, 1]."""
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    P_norm = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-8)
    return X_norm @ P_norm.T


################======== Experiment Logic ========################
def run_single_experiment(random_state):
    # Reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    
    # Data Prep & Splitting
    transform = transforms.ToTensor()
    raw_ds = ConcatDataset([
        datasets.MNIST(root='./data', train=True, transform=transform, download=True),
        datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    ])
    
    # Count labels
    all_targets = np.concatenate([d.targets.numpy() for d in raw_ds.datasets])
    unique_labels = np.unique(all_targets)    
    class_indices = [np.where(all_targets == i)[0] for i in unique_labels]
    min_count = min(len(idxs) for idxs in class_indices)
    
    # Subsample indices to ensure label balance
    balanced_indices = []
    rng = np.random.RandomState(random_state) 
    for idxs in class_indices:
        selected_idxs = rng.choice(idxs, min_count, replace=False)
        balanced_indices.extend(selected_idxs)
        
    ds_full = Subset(raw_ds, balanced_indices)
    targets = all_targets[balanced_indices]

    # Filter OOD samples
    id_idx = np.where(np.isin(targets, LABELS_ID))[0]
    ood_idx = np.where(np.isin(targets, LABELS_OOD))[0]
    
    ds_id = Subset(ds_full, id_idx)
    ds_ood = Subset(ds_full, ood_idx)
    
    # Split in-distribution (IDs): 80% Train, 15% Calib, 5% Test
    n_total = len(ds_id)
    n_train = int(0.8 * n_total)
    n_calib = int(0.15 * n_total)
    n_test = n_total - n_train - n_calib
    
    ds_train, ds_calib, ds_test = random_split(
        ds_id, [n_train, n_calib, n_test], 
        generator=torch.Generator().manual_seed(random_state)
    )
    
    # Loaders
    ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    ld_calib = DataLoader(ds_calib, batch_size=BATCH_SIZE, shuffle=False)
    ld_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    ld_ood = DataLoader(ds_ood, batch_size=BATCH_SIZE, shuffle=False)
    
    # HDC Init
    pos_hvs = make_position_hvs(28*28, DIM, device=DEVICE)
    
    # Train Models (Train Only - Used by both Conformal & Baselines for fair comparison)
    protos_train = build_prototypes(ld_train, pos_hvs, LABELS_ID)
    protos_train_np = protos_train.cpu().numpy()
    print("Prototypes built.")
    sys.stdout.flush()

    # Pre-compute HVs
    calib_hvs, calib_y = get_hvs_labels(ld_calib, pos_hvs)
    test_hvs, test_y = get_hvs_labels(ld_test, pos_hvs)
    ood_hvs, ood_y = get_hvs_labels(ld_ood, pos_hvs)
    print("Encoding complete.")
    sys.stdout.flush()

    # Balance OOD to match Test size 
    min_len = min(len(ood_hvs), len(test_hvs))
    ood_hvs, ood_y = ood_hvs[:min_len], ood_y[:min_len]
    
    # Initialize our Conformal Model
    chdc = ConformalHDC(protos_train_np, LABELS_ID, sim_measure="cosine", random_state=random_state) 
    
    exp_results = []
    
    # =========================================================
    #    Conformal HDC Methods (sim, discount)
    # =========================================================
    for stype in CONFORMAL_SCORES:
        chdc.compute_calib_scores(calib_hvs, calib_y, score_type=stype)
        
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
    sims_id = get_cosine_similarities(test_hvs, protos_train_np)
    sims_ood = get_cosine_similarities(ood_hvs, protos_train_np)
    
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
    # Standard formula: E(x) = -log(sum(exp(sims))). 
    # Lower energy indicates ID data. For AUROC (where higher score = ID), we use -E(x)
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
        print("Usage: python ood_mnist.py <seed_group_id>")
        sys.exit(1)

    seed_arg = int(sys.argv[1])

    # Directory Setup
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
            print(f"Error in state {current_state}: {e}")
            sys.stdout.flush()

    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)
        final_df.to_csv(outfile, index=False)
        print(f"\nOOD Results saved to {outfile}")
        sys.stdout.flush()
    else:
        print("No results generated.")
        sys.stdout.flush()
