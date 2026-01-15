import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append('../') 
try:
    from NeuroHDC.FHRR import *
    from NeuroHDC.fn import *
    from conformalHDC.models import *
    from conformalHDC.methods import *
    from conformalHDC.utils import *
except ImportError:
    print("Warning: NeuroHDC or conformalHDC modules not found. Ensure '../' is in path.")


# Fixed Constants
EXP_NAME = "odor_decoding"
#REPETITIONS = 2
REPETITIONS = 100
DIM = 15_000
BETA = 0.3
SCORE_TYPES = ["sim", "ratio", "discount", "penalized", "inverse_quantile"]
TRAINING_WINDOW = (200, 600)
RUNNING_WINDOW = (0, 200)
BIN_SIZE = 25
SLICING_WINDOW = 200
STEP_BINS = 2


def run_single_experiment(random_state, rat_id, alpha, in_path_id, in_path_ood):
    # Data Loading (Function imported from NeuroHDC.fn)
    splits = prep_loader_slicing(
        irat=rat_id, 
        split_ratio=(0.5, 0.4, 0.1), 
        in_path=in_path_id,
        step_bins=STEP_BINS,
        slicing_window=SLICING_WINDOW,
        bin_size=BIN_SIZE,
        seed=random_state
    )

    X_train, y_train = splits.train.X, splits.train.y
    X_cal, y_cal = splits.cal.X, splits.cal.y
    X_test, y_test = splits.test.X, splits.test.y
    
    # Load OOD Data
    with in_path_ood.open("rb") as f:
        run_data = pickle.load(f)
    X_ood = run_data[rat_id]['binned_spk']
    print(f"Data loaded.")
    sys.stdout.flush()

    # Encoder Setup
    n, nT, p = X_train.shape 
    rff = RFF(n_feature=p, dimension=DIM, seed=random_state)
    W = rff.gen_basis(cov=np.eye(p)) 
    TB = rff.gen_time_base()
    
    # Encode
    enc_train = rff.encode_all(W, X_train, TB, beta=BETA)
    enc_cal   = rff.encode_all(W, X_cal, TB, beta=BETA)
    enc_test  = rff.encode_all(W, X_test, TB, beta=BETA)
    enc_ood   = rff.encode_all(W, X_ood, TB, beta=BETA)
    
    # Prototypes & ConformalHDC
    proto_dict = rff.build_class_prototypes(enc_train, y_train)
    unique_labels = sorted(np.unique(y_train))
    proto_matrix = np.stack([proto_dict[k] for k in unique_labels])
    print("Prototypes built.")
    sys.stdout.flush()
    
    chdc = ConformalHDC(class_HVs=proto_matrix, class_labels=unique_labels, sim_measure="complex_cosine")
    
    exp_results = []

    # Experiment Loop per Score Type
    for stype in SCORE_TYPES:
        chdc.compute_calib_scores(enc_cal, y_cal, score_type=stype)
        
        # 1. Set-Valued Prediction
        for marginal in [True, False]:
            sets = chdc.set_valued_CP(enc_test, alpha, marginal=marginal)
            sizes = [len(p) for p in sets]
            covered = [1 if y in p else 0 for y, p in zip(y_test, sets)]

            # Label conditional coverage
            lc_covs = []
            for lbl in unique_labels:
                lbl_idx = np.where(y_test == lbl)[0]
                if len(lbl_idx) > 0:
                    lc_covs.append(np.mean([covered[i] for i in lbl_idx]))

            exp_results.append({
                "exp": "set_valued",
                "random_state": random_state,
                "rat_id": rat_id,
                "score_type": stype,
                "alpha": alpha,
                "marginal": marginal,
                "set_cov": np.mean(covered),
                "set_size": np.mean(sizes),
                "min_class_cov": np.min(lc_covs) if lc_covs else 0.0,
                # Placeholders
                "point_acc": np.nan, "ood_auroc": np.nan, "method": np.nan
            })
        
        # 2. Point-Valued Prediction
        for method in ["accurate", "efficient"]:
            preds_pt = chdc.point_valued_CP(enc_test, method=method)
            acc_pt = accuracy_score(y_test, preds_pt)

            exp_results.append({
                "exp": "point_valued",
                "random_state": random_state,
                "rat_id": rat_id,
                "score_type": stype,
                "alpha": alpha,
                "method": method,
                "point_acc": acc_pt,
                # Placeholders
                "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
                "min_class_cov": np.nan, "ood_auroc": np.nan
            })

        # 3. OOD Detection
        for marginal in [True, False]:
            p_vals_id  = chdc.get_max_p_value(enc_test, marginal=marginal)
            p_vals_ood = chdc.get_max_p_value(enc_ood, marginal=marginal)
            
            y_true_roc = np.concatenate([np.ones(len(p_vals_id)), np.zeros(len(p_vals_ood))])
            y_scores_roc = np.concatenate([p_vals_id, p_vals_ood])
            ood_auroc = roc_auc_score(y_true_roc, y_scores_roc)
            
            exp_results.append({
                "exp": "ood",
                "random_state": random_state,
                "rat_id": rat_id,
                "score_type": stype,
                "alpha": alpha,
                "marginal": marginal,
                "ood_auroc": ood_auroc,
                # Placeholders
                "set_cov": np.nan, "set_size": np.nan, "point_acc": np.nan, 
                "min_class_cov": np.nan, "method": np.nan
            })
    print("Finished running conformaHDC.")
    sys.stdout.flush()

    # Baseline Vanilla HDC (Once per seed)
    preds_vanilla = chdc.predict(enc_test)
    acc_vanilla = accuracy_score(y_test, preds_vanilla)
    
    exp_results.append({
        "exp": "point_valued",
        "random_state": random_state,
        "rat_id": rat_id,
        "score_type": "cosine",
        "alpha": np.nan,
        "method": "vanilla",
        "point_acc": acc_vanilla,
        # Placeholders
        "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
        "min_class_cov": np.nan, "ood_auroc": np.nan
    })
    print("Finished running vanilla HDC.")
    sys.stdout.flush()

    return pd.DataFrame(exp_results)



# ---------------
# Main Execution
# ---------------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python exp_rat.py <seed> <rat_id> <alpha>")
        sys.exit(1)

    seed_arg = int(sys.argv[1])
    rat_arg = int(sys.argv[2])
    alpha_arg = float(sys.argv[3])

    # Paths
    path_id = Path("../data/rat") / f"odor_prep_{TRAINING_WINDOW}_{BIN_SIZE}.pickle"
    path_ood = Path("../data/rat") / f"run_prep_{RUNNING_WINDOW}_{BIN_SIZE}.pickle"

    # Directory Setup
    out_dir = Path(f"./results/{EXP_NAME}")
    out_dir.mkdir(parents=True, exist_ok=True)
    # rat_name = ['Barat','Buchanan','Mitt','Stella','Superchris']
    outfile = out_dir / f"rat{rat_arg}_seed{seed_arg}_alpha{alpha_arg}.csv"  

    print(f"Starting job: Rat {rat_arg}, Seed {seed_arg}, Alpha {alpha_arg}, Reps {REPETITIONS}")

    results_list = []
    
    for i in tqdm(range(1, REPETITIONS + 1), desc="Repetitions"):
        # Unique random state per repetition
        current_state = REPETITIONS * (seed_arg - 1) + i
        
        try:
            df_rep = run_single_experiment(current_state, rat_arg, alpha_arg, path_id, path_ood)
            results_list.append(df_rep)
        except Exception as e:
            print(f"Error in state {current_state}: {e}")

    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)
        final_df.to_csv(outfile, index=False)
        print(f"\nResults saved to {outfile}")
    else:
        print("No results generated.")
