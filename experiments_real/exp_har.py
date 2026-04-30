import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
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
EXP_NAME = "uci_har"
REPETITIONS = 10
DIM = 10_000
LEVELS = 21
BATCH_SIZE = 512
SCORE_TYPES = ["sim", "ratio", "discount", "penalized", "inverse_quantile"]

# UCI HAR Splits: 0, 1, 2 (Dynamic) as ID; 3, 4, 5 (Static) as OOD
LABELS_ID = [0, 1, 2]
LABELS_OOD = [3, 4, 5]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################======== Utility Functions (UCI HAR) ========################

@torch.no_grad()
def bipolar_sign(x):
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

def _rand_bip(shape, device=None):
    device = device or DEVICE
    r = torch.randint(0, 2, shape, device=device, dtype=torch.int8)
    return r.float().mul_(2).sub_(1)

def make_im_cim(F, Levels, D, device=None):
    device = device or DEVICE
    iM = _rand_bip((F, D), device=device)
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
    Xmin, Xmax = X.min(axis=0), X.max(axis=0)
    rng = np.maximum(Xmax - Xmin, 1e-8)
    Z = (X - Xmin) / rng
    return np.clip(np.round(Z * (levels - 1)), 0, levels - 1).astype(np.int64)

def build_prototypes_np(hvs, labels, class_list, dim):
    protos = np.zeros((len(class_list), dim))
    for i, lbl in enumerate(class_list):
        idx = np.where(labels == lbl)[0]
        if len(idx) > 0:
            protos[i] = np.sign(np.sum(hvs[idx], axis=0))
    return protos

def load_har_data():
    """ 
    Expects data in ../data/UCI_HAR/
    """
    path = "../data/UCI_HAR/"
    X_train = pd.read_csv(path + "train/X_train.txt", sep='\s+', header=None).values
    y_train = pd.read_csv(path + "train/y_train.txt", header=None).values.flatten() - 1
    X_test = pd.read_csv(path + "test/X_test.txt", sep='\s+', header=None).values
    y_test = pd.read_csv(path + "test/y_test.txt", header=None).values.flatten() - 1
    return np.vstack([X_train, X_test]), np.concatenate([y_train, y_test])


################======== Experiment Logic ========###############
def run_single_experiment(random_state, alpha):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    
    X_all, y_all = load_har_data()
    X_quant = quantize_to_levels(X_all, LEVELS)
    print("Data loading and quantization complete.")
    sys.stdout.flush()
    
    id_mask = np.isin(y_all, LABELS_ID)
    ood_mask = np.isin(y_all, LABELS_OOD)
    
    X_id, y_id = X_quant[id_mask], y_all[id_mask]
    X_ood, y_ood = X_quant[ood_mask], y_all[ood_mask]
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_id, y_id, test_size=0.1, random_state=random_state, stratify=y_id
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.4, random_state=random_state, stratify=y_train_full
    )
    
    def to_loader(X_np, y_np):
        return DataLoader(TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np)), 
                          batch_size=BATCH_SIZE, shuffle=False)

    ld_train = to_loader(X_train, y_train)
    ld_cal   = to_loader(X_cal, y_cal)
    ld_test  = to_loader(X_test, y_test)
    ld_ood   = to_loader(X_ood, y_ood)
    
    F = X_all.shape[1]
    iM, CiM = make_im_cim(F, LEVELS, DIM, device=DEVICE)

    # Measure Encoding Time
    start_enc = time.time()
    train_hvs, train_y = get_hvs_labels(ld_train, iM, CiM)
    cal_hvs, cal_y     = get_hvs_labels(ld_cal, iM, CiM)
    encoding_train = time.time() - start_enc
    
    start_enc = time.time()
    test_hvs, test_y   = get_hvs_labels(ld_test, iM, CiM)
    encoding_test = time.time() - start_enc
    
    ood_hvs, ood_y     = get_hvs_labels(ld_ood, iM, CiM)
    print("Encoding complete.")
    sys.stdout.flush()
    
    min_len = min(len(ood_hvs), len(test_hvs))
    ood_hvs, ood_y = ood_hvs[:min_len], ood_y[:min_len]

    # Measure Training Time (Prototype Building)
    start_train = time.time()
    protos_train = build_prototypes_np(train_hvs, train_y, LABELS_ID, DIM)
    training_time = time.time() - start_train
    
    full_hvs, full_y = np.concatenate([train_hvs, cal_hvs]), np.concatenate([train_y, cal_y])
    protos_full = build_prototypes_np(full_hvs, full_y, LABELS_ID, DIM)
    print("Prototypes built.")
    sys.stdout.flush()

    chdc = ConformalHDC(protos_train, LABELS_ID, sim_measure="cosine", random_state=random_state)
    exp_results = []
    runtime_results = []
    
    for stype in SCORE_TYPES:
        # Calibrate the set-valued model and measure overhead
        start_calib = time.time()
        chdc.compute_calib_scores(cal_hvs, cal_y, score_type=stype)
        calib_overhead = time.time() - start_calib

        # 1. Set-Valued Prediction 
        for marginal in [True, False]:
            # Measure Inference/Set-Generation Overhead
            start_inf = time.time()
            sets = chdc.set_valued_CP(test_hvs, alpha, marginal=marginal)
            inference_overhead = time.time() - start_inf
            sizes = [len(p) for p in sets]
            covered = [1 if y in p else 0 for y, p in zip(test_y, sets)]
            lc_covs = [np.mean([covered[i] for i in np.where(test_y == lbl)[0]]) for lbl in LABELS_ID]
            
            exp_results.append({
                "exp": "set_valued", 
                "random_state": random_state, 
                "score_type": stype,
                "alpha": alpha,
                "marginal": marginal,
                "set_cov": np.mean(covered), 
                "set_size": np.mean(sizes),
                "lc_covs": lc_covs, 
                # Placeholders
                "point_acc": np.nan, "lc_accs":np.nan, "ood_auroc": np.nan
            })

            runtime_results.append({
            "exp": "set_valued", 
            "random_state": random_state,
            "score_type": stype,
            "calib_overhead": calib_overhead,
            "inference_overhead": inference_overhead,
            "marginal": marginal,
            })
        
        # 2. Point-Valued Prediction
        start_inf = time.time()
        preds_pt = chdc.point_valued_CP(test_hvs, alpha, allow_empty=False, marginal=False)
        inference_overhead = time.time() - start_inf
        
        exp_results.append({
            "exp": "point_valued", 
            "random_state": random_state, 
            "score_type": stype, 
            "alpha": alpha,
            "point_acc": eval_accuracy(np.array(preds_pt).ravel(), test_y),
            "lc_accs": eval_lc_accuracy(preds_pt, test_y, LABELS_ID),
            # Placeholders
            "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, "lc_covs": np.nan, "ood_auroc": np.nan
        })

        runtime_results.append({
            "exp": "point_valued", 
            "random_state": random_state,
            "score_type": stype,
            "calib_overhead": calib_overhead,
            "inference_overhead": inference_overhead,
            "marginal": np.nan
        })
        
        # 3. OOD Detection
        for marginal in [True, False]:
            p_id, p_ood = chdc.get_max_p_value(test_hvs, marginal=marginal), chdc.get_max_p_value(ood_hvs, marginal=marginal)
            exp_results.append({
                "exp": "ood", 
                "random_state": random_state, 
                "score_type": stype, 
                "alpha": alpha,
                "marginal": marginal, 
                "ood_auroc": roc_auc_score(np.concatenate([np.ones(len(p_id)), np.zeros(len(p_ood))]), np.concatenate([p_id, p_ood])),
                # Placeholders
                "set_cov": np.nan, "set_size": np.nan, "point_acc": np.nan, "lc_covs": np.nan, "lc_accs":np.nan
            })
            
    # Baseline Vanilla
    for p_set, s_name in [(protos_train, "vanilla_train"), (protos_full, "vanilla_full")]:
        v_model = ConformalHDC(p_set, LABELS_ID, sim_measure="cosine", random_state=random_state)
        start_inf = time.time()
        preds = v_model.predict(test_hvs)
        hdc_inference = time.time() - start_inf
        exp_results.append({
            "exp": "point_valued", 
            "random_state": random_state, 
            "score_type": s_name, 
            "alpha": alpha,
            "point_acc": eval_accuracy(preds, test_y), "lc_accs": eval_lc_accuracy(preds, test_y, LABELS_ID),
            # Placeholders
            "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, "lc_covs": np.nan, "ood_auroc": np.nan
        })

    # Add shared Training/Encoding times as a final row
    df_runtime = pd.DataFrame(runtime_results)
    df_runtime['encoding_test'] = encoding_test
    df_runtime['training_time'] = training_time + encoding_train
    df_runtime['hdc_inference'] = hdc_inference
    df_runtime['n_test'] = len(test_hvs)
    
    print("Finished running ConformalHDC and Vanilla.")
    sys.stdout.flush()
    return pd.DataFrame(exp_results), df_runtime


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python exp_uci_har.py <seed_group_id> <alpha>")
        sys.exit(1)

    seed_arg, alpha_arg = int(sys.argv[1]), float(sys.argv[2])
    out_dir = Path(f"./results/{EXP_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"seed{seed_arg}_alpha{alpha_arg}.csv"
    outfile_runtime = out_dir / f"seed{seed_arg}_alpha{alpha_arg}_runtime.csv"

    results_list = []
    runtime_list = []

    for i in tqdm(range(1, REPETITIONS + 1), desc="Repetitions"):
        current_state = REPETITIONS * (seed_arg - 1) + i
        try:
            df_rep, df_runtime = run_single_experiment(current_state, alpha_arg)
            results_list.append(df_rep)
            runtime_list.append(df_runtime)
        except Exception as e:
            print(f"Error in state {current_state}: {e}")

    if results_list:
        pd.concat(results_list, ignore_index=True).to_csv(outfile, index=False)
        print(f"\nResults saved to {outfile}")

    if runtime_list:
        pd.concat(runtime_list, ignore_index=True).to_csv(outfile_runtime, index=False)
        print(f"Runtime results saved to {outfile_runtime}")