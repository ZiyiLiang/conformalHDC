import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, roc_curve

# --- Library Imports ---
sys.path.append('../') 
try:
    from conformalHDC.models import *
    from conformalHDC.methods import *
    from conformalHDC.utils import *
except ImportError:
    print("Warning: conformalHDC modules not found. Ensure '../' is in path.")


# Fixed Constants
EXP_NAME = "mnist"
REPETITIONS = 10
DIM = 10_000
BATCH_SIZE = 512
LABELS_ID = [0, 1, 2, 3, 4, 5]
LABELS_OOD = [6, 7, 8, 9]
SCORE_TYPES = ["sim", "ratio", "discount", "penalized", "inverse_quantile"]

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

@torch.no_grad()
def cosine_sim(a, b):
    a, b = a.float(), b.float()
    an = torch.norm(a, dim=1, keepdim=True).clamp_min(1e-8)
    bn = torch.norm(b, dim=1, keepdim=True).clamp_min(1e-8).T
    return (a @ b.T) / (an * bn)

def get_hvs_labels(loader, pos_hvs):
    all_hvs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(pos_hvs.device)
        hvs = encode_binary_images_to_hvs(imgs, pos_hvs)
        all_hvs.append(hvs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_hvs), np.concatenate(all_labels)


################======== Experiment Logic ========################

def run_single_experiment(random_state, alpha):
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
    
    # Train Models
    # 1. Train Only (For ConformalHDC)
    protos_train = build_prototypes(ld_train, pos_hvs, LABELS_ID)
    
    # 2. Full (Train + Calib) (For Vanilla)
    ds_full_id = ConcatDataset([ds_train, ds_calib])
    ld_full = DataLoader(ds_full_id, batch_size=BATCH_SIZE, shuffle=True)
    protos_full = build_prototypes(ld_full, pos_hvs, LABELS_ID)
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
    
    chdc = ConformalHDC(protos_train.cpu().numpy(), LABELS_ID, sim_measure="cosine", random_state=random_state) 
    
    exp_results = []
    
    # --- Experiment Loop per Score Type ---
    for stype in SCORE_TYPES:
        chdc.compute_calib_scores(calib_hvs, calib_y, score_type=stype)
        
        # 1. Set-Valued Prediction
        for marginal in [True, False]:
            sets = chdc.set_valued_CP(test_hvs, alpha, marginal=marginal)
            sizes = [len(p) for p in sets]
            covered = [1 if y in p else 0 for y, p in zip(test_y, sets)]
            
            # Label conditional coverage
            lc_covs = []
            for lbl in LABELS_ID:
                lbl_idx = np.where(test_y == lbl)[0]
                if len(lbl_idx) > 0:
                    lc_covs.append(np.mean([covered[i] for i in lbl_idx]))
            
            exp_results.append({
                "exp": "set_valued",
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
        lc_accs = eval_lc_accuracy(preds_pt, test_y, LABELS_ID)

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
    lc_accs = eval_lc_accuracy(preds_vanilla, test_y, LABELS_ID)
    
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

    chdc_full = ConformalHDC(protos_full.cpu().numpy(), LABELS_ID, sim_measure="cosine", random_state=random_state)
    preds_vanilla = chdc_full.predict(test_hvs)
    acc_vanilla = eval_accuracy(preds_vanilla, test_y)
    lc_accs = eval_lc_accuracy(preds_vanilla, test_y, LABELS_ID)
    
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
        print("Usage: python exp_mnist.py <seed_group_id> <alpha>")
        sys.exit(1)

    seed_arg = int(sys.argv[1])
    alpha_arg = float(sys.argv[2])

    # Directory Setup
    out_dir = Path(f"./results/{EXP_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    outfile = out_dir / f"seed{seed_arg}_alpha{alpha_arg}.csv"

    print(f"Starting job: Seed Group {seed_arg}, Alpha {alpha_arg}, Reps {REPETITIONS}")
    sys.stdout.flush()

    results_list = []

    for i in tqdm(range(1, REPETITIONS + 1), desc="Repetitions"):
        # Generate unique seed based on group ID and repetition index
        print(f"Running repetition {i}...")
        sys.stdout.flush()
        current_state = REPETITIONS * (seed_arg - 1) + i
        
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
