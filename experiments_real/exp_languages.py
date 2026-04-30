import sys
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Subset, DataLoader, random_split
from torchhd import functional, embeddings
from torchhd.datasets import EuropeanLanguages as Languages
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Library Imports ---
sys.path.append('../') 
try:
    from conformalHDC.models import *
    from conformalHDC.methods import *
    from conformalHDC.utils import *
except ImportError:
    print("Warning: conformalHDC modules not found. Ensure '../' is in path.")

# Fixed Constants
EXP_NAME = "languages"
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

# OOD (Implicit): The Uralic Family (Non-Indo-European)
# 'Finnish', 'Estonian', 'Hungarian'

SCORE_TYPES = ["sim", "ratio", "discount", "penalized", "inverse_quantile"]
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

################======== Experiment Logic ========################

def run_single_experiment(random_state, alpha):
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

    for cls_idx in range(len(all_class_names)):
        # Get all indices for this specific class
        cls_indices = np.where(all_targets == cls_idx)[0]
        np.random.shuffle(cls_indices) 
        # Take up to the limit
        subsampled_indices.extend(cls_indices[:MAX_SAMPLES_PER_CLASS])
    
    subsample_mask = np.zeros(len(all_targets), dtype=bool)
    subsample_mask[subsampled_indices] = True

    # Filter ID vs OOD using the subsampled mask
    id_indices = np.where(np.isin(all_targets, ID_LABELS) & subsample_mask)[0]
    ood_indices = np.where(~np.isin(all_targets, ID_LABELS) & subsample_mask)[0]
    
    ds_id_full = Subset(ds_full, id_indices)
    ds_ood = Subset(ds_full, ood_indices)
    
    # Random Split
    n_total = len(ds_id_full)
    n_train = int(0.75 * n_total)
    n_calib = int(0.225 * n_total)
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
    print(f"Train size per class: {int(len(ds_train)/len(ID_LANG_STRINGS))}, " +
        f"calib size per class: {int(len(ds_calib)/len(ID_LANG_STRINGS))}, " +
        f"test size per class: {int(len(ds_test)/len(ID_LANG_STRINGS))}\n")
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

    # Build Prototypes (FLOAT)
    # 1. Train Only (For ConformalHDC)
    protos_train = build_prototypes_float(train_hvs, train_y, ID_LABELS, DIMENSIONS)
    
    # 2. Full (Train + Calib) (For Vanilla)
    full_hvs = np.concatenate([train_hvs, cal_hvs])
    full_y = np.concatenate([train_y, cal_y])
    protos_full = build_prototypes_float(full_hvs, full_y, ID_LABELS, DIMENSIONS)
    print("Prototypes built.")
    sys.stdout.flush()

    # Conformal Prediction
    # We pass ID_LABELS so the class maps the ith prototype to the correct real label
    chdc  = ConformalHDC(protos_train, ID_LABELS, sim_measure="cosine", random_state=random_state)
    
    exp_results = []
    
    for stype in SCORE_TYPES:
        chdc.compute_calib_scores(cal_hvs, cal_y, score_type=stype)
        
        # 1. Set-Valued Prediction
        for marginal in [True, False]:
            sets = chdc.set_valued_CP(test_hvs, alpha, marginal=marginal)
            sizes = [len(p) for p in sets]
            covered = [1 if y in p else 0 for y, p in zip(test_y, sets)]
            
            # Label conditional coverage
            lc_covs = []
            for lbl in ID_LABELS:
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
                "min_class_cov": lc_covs if lc_covs else 0.0,
                # Placeholders
                "point_acc": np.nan, "lc_accs":np.nan, "ood_auroc": np.nan
            })

        # 2. Point-Valued Prediction
        preds_pt = chdc.point_valued_CP(test_hvs, alpha, allow_empty=False, marginal=False)
        preds_pt = np.array(preds_pt).ravel()
        acc_pt = accuracy_score(preds_pt, test_y)
        lc_accs = eval_lc_accuracy(preds_pt, test_y, ID_LABELS)
        
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
            p_in = chdc.get_max_p_value(test_hvs, marginal=marginal)
            p_ood = chdc.get_max_p_value(ood_hvs, marginal=marginal)
            
            y_roc = np.concatenate([np.ones(len(p_in)), np.zeros(len(p_ood))])
            s_roc = np.concatenate([p_in, p_ood])
            auroc = roc_auc_score(y_roc, s_roc)
            
            exp_results.append({
                "exp": "ood",
                "random_state": random_state,
                "score_type": stype,
                "alpha": alpha,
                "marginal": marginal,
                "ood_auroc": auroc,
                # Placeholders
                "set_cov": np.nan, "set_size": np.nan, "point_acc": np.nan, 
                "lc_covs": np.nan, "lc_accs": np.nan
            })
            
    print("Finished running ConformalHDC.")
    sys.stdout.flush()

    # Baseline Vanilla HDC (Train Only)
    preds_vanilla = chdc.predict(test_hvs)
    acc_vanilla = accuracy_score(preds_vanilla, test_y)
    lc_accs = eval_lc_accuracy(preds_vanilla, test_y, ID_LABELS)
    
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

    # Baseline Vanilla HDC (Full Train+Cal)
    vanilla_full = ConformalHDC(protos_train, ID_LABELS, 
                                sim_measure="cosine", random_state=random_state)
    preds_vanilla_full = vanilla_full.predict(test_hvs)
    acc_vanilla_full = accuracy_score(preds_vanilla_full, test_y)
    lc_accs_full = eval_lc_accuracy(preds_vanilla_full, test_y, ID_LABELS)
    
    exp_results.append({
        "exp": "point_valued",
        "random_state": random_state,
        "score_type": "vanilla_full",
        "alpha": alpha,
        "point_acc": acc_vanilla_full,
        "lc_accs": lc_accs_full,
        # Placeholders
        "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
        "lc_covs": np.nan, "ood_auroc": np.nan
    })

    print("Finished running vanilla HDC.")
    sys.stdout.flush()
    
    return pd.DataFrame(exp_results)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python exp_languages.py <seed_group_id> <alpha>")
        sys.exit(1)

    seed_arg = int(sys.argv[1])
    alpha_arg = float(sys.argv[2])

    out_dir = Path(f"./results/{EXP_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"seed{seed_arg}_alpha{alpha_arg}.csv"

    print(f"Starting job: Seed Group {seed_arg}, Alpha {alpha_arg}, Reps {REPETITIONS}")
    
    results = []
    for i in tqdm(range(1, REPETITIONS + 1)):
        state = REPETITIONS * (seed_arg - 1) + i
        try:
            results.append(run_single_experiment(state, alpha_arg))
        except Exception as e:
            print(f"Error state {state}: {e}")
            
    if results:
        pd.concat(results).to_csv(outfile, index=False)
        print("Done.")
