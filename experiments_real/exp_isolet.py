import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Library Imports ---
# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.load import load_isolet_data
try:
    from conformalHDC.models import *
    from conformalHDC.methods import *
    from conformalHDC.utils import *
    from conformalHDC.jackknife import JackknifePlusHDC
    from conformalHDC.fcp import FullConformalHDC
except ImportError:
    print("Warning: conformalHDC modules not found. Ensure '../' is in path.")


# Fixed Constants
EXP_NAME = "isolet"
REPETITIONS = 10
DIM = 10_000
LEVELS = 21
BATCH_SIZE = 512
SCORE_TYPES = ["sim", "ratio", "discount", "penalized", "inverse_quantile"]

# ISOLET Splits
ID_CLASSES = list(range(23))
OOD_CLASSES = list(range(23, 26))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



################======== Utility Functions (ISOLET) ========################
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


################======== Experiment Logic ========################
def run_single_experiment(random_state, alpha):
    # Reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    
    # Data Loading (Fetch once, split internally)
    X, y_int = load_isolet_data()
    
    # Quantize
    L_all = quantize_to_levels(X, LEVELS)
    print("Data loading and quantizatization complete.")
    sys.stdout.flush()
    
    # ID / OOD Split
    id_mask = np.isin(y_int, ID_CLASSES)
    ood_mask = np.isin(y_int, OOD_CLASSES)
    
    X_id, y_id = L_all[id_mask], y_int[id_mask]
    X_ood, y_ood = L_all[ood_mask], y_int[ood_mask]
    
    # Train/Cal/Test Split on ID data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_id, y_id, test_size=0.05, random_state=random_state, stratify=y_id
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.4, random_state=random_state, stratify=y_train_full
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

    # Build Prototypes
    # A. Train Only (For Calibration & Set-Valued validity)
    protos_train = build_prototypes_np(train_hvs, train_y, ID_CLASSES, DIM)
    
    # B. Full (Train + Cal) (For Point-Valued efficiency & Vanilla)
    full_hvs = np.concatenate([train_hvs, cal_hvs])
    full_y = np.concatenate([train_y, cal_y])
    protos_full = build_prototypes_np(full_hvs, full_y, ID_CLASSES, DIM)
    print("Prototypes built.")
    sys.stdout.flush()
    def prototype_builder(HVs, labels, class_labels):
            return build_prototypes_np(HVs, labels, class_labels, HVs.shape[1])

    # Initialize Models
    chdc = ConformalHDC(protos_train, ID_CLASSES, sim_measure="cosine", random_state=random_state)

    jknife = JackknifePlusHDC(
        ID_CLASSES, prototype_builder,
        sim_measure="cosine", random_state=chdc.random_state,
    )

    fcp = FullConformalHDC(
        ID_CLASSES, prototype_builder,
        sim_measure="cosine", random_state=chdc.random_state,
    )
    
    exp_results = []
    
    for stype in SCORE_TYPES:

        # Calibrate the set-valued model
        chdc.compute_calib_scores(cal_hvs, cal_y, score_type=stype)
        jknife.fit(full_hvs, full_y, score_type=stype) 
        fcp.fit(full_hvs, full_y, score_type=stype)
        
        # 1. Set-Valued Prediction 
        for method, marginal, model in [
            ("split_conformal", True, chdc),
            ("split_conformal", False, chdc),
            ("jackknife_plus", True, jknife),
            ("full_conformal", True, fcp),
        ]:
            if model is chdc:
                sets = model.set_valued_CP(test_hvs, alpha, marginal=marginal)
            else:
                sets = model.set_valued_CP(test_hvs, alpha)
     
            sizes = [len(p) for p in sets]
            covered = np.array([
                y in pset for y, pset in zip(test_y, sets)
            ])
                
            # Label conditional coverage
            lc_covs = [
                np.mean(covered[test_y == lbl])
                for lbl in ID_CLASSES if np.any(test_y == lbl)
            ]
            exp_results.append({
                "exp": "set_valued",
                "method": method,
                "random_state": random_state,
                "score_type": stype,
                "alpha": alpha,
                "marginal": marginal,
                "set_cov": np.mean(covered),
                "set_size": np.mean(sizes),
                "lc_covs": lc_covs if lc_covs else 0.0,
                # Placeholders
                "point_acc": np.nan, "lc_accs":np.nan, "ood_auroc": np.nan
            })

        # 2. Point-Valued Prediction
        preds_pt = chdc.point_valued_CP(test_hvs, alpha, allow_empty=False, marginal=False)
        acc_pt = eval_accuracy(np.array(preds_pt).ravel(), test_y)
        lc_accs = eval_lc_accuracy(preds_pt, test_y, ID_CLASSES)

        exp_results.append({
            "exp": "point_valued",
            "random_state": random_state,
            "score_type": stype,
            "alpha": alpha,
            "point_acc": acc_pt,
            "lc_accs": lc_accs,
            # Placeholders
            "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
            "lc_covs": np.nan, "ood_auroc": np.nan
        })

        # 3. OOD Detection
        for marginal in [True, False]:
            p_vals_id = chdc.get_max_p_value(test_hvs, marginal=marginal)
            p_vals_ood = chdc.get_max_p_value(ood_hvs, marginal=marginal)
            
            y_true_roc = np.concatenate([np.ones(len(p_vals_id)), np.zeros(len(p_vals_ood))])
            y_scores_roc = np.concatenate([p_vals_id, p_vals_ood])
            
            ood_auroc = roc_auc_score(y_true_roc, y_scores_roc)
            
            exp_results.append({
                "exp": "ood",
                "random_state": random_state,
                "score_type": stype,
                "alpha": alpha,
                "marginal": marginal,
                "ood_auroc": ood_auroc,
                # Placeholders
                "set_cov": np.nan, "set_size": np.nan, "point_acc": np.nan, 
                "lc_covs": np.nan, "lc_accs":np.nan
            })
            
    print("Finished running ConformalHDC.")
    sys.stdout.flush()

    # Baseline Vanilla HDC (Once per seed)
    preds_vanilla = chdc.predict(test_hvs)
    acc_vanilla = eval_accuracy(preds_vanilla, test_y)
    lc_accs = eval_lc_accuracy(preds_vanilla, test_y, ID_CLASSES)
    
    exp_results.append({
        "exp": "point_valued",
        "random_state": random_state,
        "score_type": "vanilla_train",
        "alpha": alpha,
        "point_acc": acc_vanilla, 
        "lc_accs": lc_accs,
        # Placeholders
        "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
        "lc_covs": np.nan, "ood_auroc": np.nan
    })
    
    chdc_full = ConformalHDC(protos_full, ID_CLASSES, sim_measure="cosine", random_state=random_state)
    preds_vanilla = chdc_full.predict(test_hvs)
    acc_vanilla = eval_accuracy(preds_vanilla, test_y)
    lc_accs = eval_lc_accuracy(preds_vanilla, test_y, ID_CLASSES)
    
    exp_results.append({
        "exp": "point_valued",
        "random_state": random_state,
        "score_type": "vanilla_full",
        "alpha": alpha,
        "point_acc": acc_vanilla, 
        "lc_accs": lc_accs,
        # Placeholders
        "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
        "lc_covs": np.nan, "ood_auroc": np.nan
    })
    
    print("Finished running vanilla HDC.")
    sys.stdout.flush()

    return pd.DataFrame(exp_results)


# ---------------
# Main Execution
# ---------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python exp_isolet.py <seed_group_id> <alpha>")
        sys.exit(1)

    seed_arg = int(sys.argv[1])
    alpha_arg = float(sys.argv[2])

    # Directory Setup
    out_dir = Path(f"./results/{EXP_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    outfile = out_dir / f"seed{seed_arg}_alpha{alpha_arg}.csv"

    print(f"Starting job: Seed Group {seed_arg}, Alpha {alpha_arg}, Reps {REPETITIONS}")

    results_list = []

    for i in tqdm(range(1, REPETITIONS + 1), desc="Repetitions"):
        current_state = REPETITIONS * (seed_arg - 1) + i
        print(f"Running repetition {i}...")
        sys.stdout.flush()
        
        try:
            df_rep = run_single_experiment(current_state, alpha_arg)
            results_list.append(df_rep)
        except Exception as e:
            print(f"Error in state {current_state}: {e}")
            sys.stdout.flush()

    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)
        final_df.to_csv(outfile, index=False)
        print(f"\nResults saved to {outfile}")
        sys.stdout.flush()
    else:
        print("No results generated.")
        sys.stdout.flush()
