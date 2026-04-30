import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
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
EXP_NAME = "ood_isolet"
REPETITIONS = 10
DIM = 10_000
LEVELS = 21
BATCH_SIZE = 512

# ISOLET Splits
ID_CLASSES = list(range(23))
OOD_CLASSES = list(range(23, 26))

CONFORMAL_SCORES = ["sim", "discount"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################======== Utility Functions ========################
@torch.no_grad()
def bipolar_sign(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

def _rand_bip(shape, device=None):
    device = device or DEVICE
    r = torch.randint(0, 2, shape, device=device, dtype=torch.int8)
    return r.float().mul_(2).sub_(1)

def make_im_cim(F, Levels, D, device=None):
    device = device or DEVICE
    # iM: Random Orthogonal
    iM = _rand_bip((F, D), device=device)
    
    # CiM: Correlated (Progressive bit flip)
    base = _rand_bip((D,), device=device)
    perm = torch.randperm(D, device=device)
    CiM = torch.empty((Levels, D), device=device, dtype=torch.float32)
    for l in range(Levels):
        k = (l * D) // max(1, Levels - 1)
        hv = base.clone()
        if k > 0:
            hv[perm[:k]] = -hv[perm[:k]]
        CiM[l] = hv
    return iM, CiM

@torch.no_grad()
def encode_batch(L_batch, iM, CiM, block_features=64):
    B, F_dim = L_batch.shape
    D = iM.shape[1]
    hv_sum = torch.zeros((B, D), device=iM.device)
    
    for f0 in range(0, F_dim, block_features):
        f1 = min(f0 + block_features, F_dim)
        vals = L_batch[:, f0:f1].long().to(iM.device)
        CiM_sel = CiM[vals] 
        iM_blk  = iM[f0:f1].unsqueeze(0) 
        hv_sum.add_((CiM_sel * iM_blk).sum(dim=1))
        
    return bipolar_sign(hv_sum)

@torch.no_grad()
def get_hvs_labels(loader, iM, CiM):
    all_hvs, all_lbls = [], []
    for X_b, y_b in loader:
        X_b = X_b.to(DEVICE)
        hvs = encode_batch(X_b, iM, CiM)
        all_hvs.append(hvs.cpu().numpy())
        all_lbls.append(y_b.numpy())
    return np.concatenate(all_hvs), np.concatenate(all_lbls)

def quantize_to_levels(X, levels=21):
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    rng  = np.maximum(Xmax - Xmin, 1e-8)
    Z = (X - Xmin) / rng
    L = np.clip(np.round(Z * (levels - 1)), 0, levels - 1).astype(np.int64)
    return L

def build_prototypes_np(hvs, labels, class_list, dim):
    protos = np.zeros((len(class_list), dim))
    for i, lbl in enumerate(class_list):
        idx = np.where(labels == lbl)[0]
        if len(idx) > 0:
            protos[i] = np.sign(np.sum(hvs[idx], axis=0))
    return protos

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
    
    # Data Loading (Fetch once, split internally)
    iso = fetch_openml('isolet', version=1, as_frame=False, parser='auto')
    X = iso['data'].astype(np.float32)
    y = iso['target']
    
    # Map labels 'A'..'Z' to 0..25
    classes = sorted(np.unique(y).tolist())
    label_to_id = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_to_id[s] for s in y], dtype=np.int64)
    
    # Quantize
    L_all = quantize_to_levels(X, LEVELS)
    print("Data loading and quantization complete.")
    sys.stdout.flush()
    
    # ID / OOD Split
    id_mask = np.isin(y_int, ID_CLASSES)
    ood_mask = np.isin(y_int, OOD_CLASSES)
    
    X_id, y_id = L_all[id_mask], y_int[id_mask]
    X_ood, y_ood = L_all[ood_mask], y_int[ood_mask]
    
    # Train/Cal/Test Split on ID data
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X_id, y_id, test_size=0.05, random_state=random_state, stratify=y_id
    )

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=(0.4), random_state=random_state, stratify=y_train_cal
    )
    
    # Loaders
    def to_loader(X_np, y_np):
        return DataLoader(TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np)), 
                          batch_size=BATCH_SIZE, shuffle=False)

    ld_train = to_loader(X_train, y_train)
    ld_cal   = to_loader(X_cal, y_cal)
    ld_test  = to_loader(X_test, y_test)
    ld_ood   = to_loader(X_ood, y_ood)
    print(f"ID: {len(X_train)} train, {len(X_cal)} calib, {len(X_test)} test. OOD: {len(X_ood)}")
    sys.stdout.flush()

    # HDC Init & Encoding
    F = X.shape[1]
    iM, CiM = make_im_cim(F, LEVELS, DIM, device=DEVICE)
    
    train_hvs, train_y = get_hvs_labels(ld_train, iM, CiM)
    cal_hvs, cal_y     = get_hvs_labels(ld_cal, iM, CiM)
    test_hvs, test_y   = get_hvs_labels(ld_test, iM, CiM)
    ood_hvs, ood_y     = get_hvs_labels(ld_ood, iM, CiM)
    print("Encoding complete.")
    sys.stdout.flush()
    
    # Balance OOD to match Test size
    min_len = min(len(ood_hvs), len(test_hvs))
    ood_hvs, ood_y = ood_hvs[:min_len], ood_y[:min_len]

    # Build Prototypes (Train Only)
    protos_train = build_prototypes_np(train_hvs, train_y, ID_CLASSES, DIM)
    print("Prototypes built.")
    sys.stdout.flush()

    # Initialize our Conformal Model
    chdc = ConformalHDC(protos_train, ID_CLASSES, sim_measure="cosine", random_state=random_state) 
    
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
        print("Usage: python ood_isolet.py <seed_group_id>")
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
