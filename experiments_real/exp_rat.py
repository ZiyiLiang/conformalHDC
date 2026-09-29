import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm 
from sklearn.metrics import accuracy_score

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from hdc_encoder.neuro import *
    from conformal_inference.models import *
    from conformal_inference.methods import *
    from conformal_inference.utils import *
    from conformal_inference.refit import (
        JackknifePlusHDC, FullConformalHDC, jackknife_all_scores,full_conformal_all_scores
    )
    from data.load import load_rat_data
    from data.config import RESULTS_ROOT
    from experiments_real.common import (
        eval_accuracy, eval_lc_accuracy, adaptive_alpha,
        development_set, set_metrics, ood_auroc, log,
    )
except ImportError:
    print("Warning: NeuroHDC or conformalHDC modules not found. Ensure '../' is in path.")


# Fixed Constants
# one repetition takes ~ 400 MB, 1 min
EXP_NAME = "odor_decoding"
REPETITIONS = 4
# REPETITIONS = 100   
DIM = 15_000
SCORE_TYPES = ["sim", "ratio", "discount", "penalized", "inverse_quantile"]
TRAINING_WINDOW = (200, 600)
RUNNING_WINDOW = (0, 200)
BIN_SIZE = 25
SLICING_WINDOW = 200
STEP_BINS = 2

def run_single_experiment(random_state, rat_id, alpha, beta):

    # Data Loading
    splits, X_ood = load_rat_data(
        irat=rat_id, 
        split_ratio=(0.5, 0.4, 0.1), 
        training_window=TRAINING_WINDOW,
        running_window=RUNNING_WINDOW,
        step_bins=STEP_BINS,
        slicing_window=SLICING_WINDOW,
        bin_size=BIN_SIZE,
        seed=random_state
    )

    X_train, y_train = splits.train.X, splits.train.y
    X_cal, y_cal = splits.cal.X, splits.cal.y
    X_test, y_test = splits.test.X, splits.test.y
    
    log("Data loaded.")

    # Encoder Setup
    n, nT, p = X_train.shape 
    rff = RFF(n_feature=p, dimension=DIM, seed=random_state)
    W = rff.gen_basis(cov=np.eye(p)) 
    TB = rff.gen_time_base()
    
    # Encode
    enc_train = rff.encode_all(W, X_train, TB, beta=beta)
    enc_cal   = rff.encode_all(W, X_cal, TB, beta=beta)
    enc_test  = rff.encode_all(W, X_test, TB, beta=beta)
    enc_ood   = rff.encode_all(W, X_ood, TB, beta=beta)
    
    # Train Models
    # 1. Train Only (For ConformalHDC)
    proto_dict = rff.build_class_prototypes(enc_train, y_train)
    unique_labels = sorted(np.unique(y_train))
    proto_train = np.stack([proto_dict[k] for k in unique_labels])

    # 2. Full (Train + Calib) (For Vanilla)
    enc_full, y_full = development_set((enc_train, y_train), (enc_cal, y_cal))
    proto_full_dict = rff.build_class_prototypes(enc_full, y_full)
    proto_full = np.stack([proto_full_dict[k] for k in unique_labels])

    log("Prototypes built.")

    # Conformal methods
    chdc = CachedConformalHDC(
        class_HVs=proto_train, class_labels=unique_labels, 
        sim_measure="complex_cosine",
    ).cache_similarities(enc_cal, enc_test, enc_ood) # re-use the similarit for every dataset
    
    def prototype_builder(HVs, labels, class_labels):
        prototypes = rff.build_class_prototypes(HVs, labels)
        # A leave-one-out subset can lose a class; use a zero prototype.
        empty = np.zeros(HVs.shape[1], dtype=HVs.dtype)
        return np.stack([prototypes.get(label, empty) for label in class_labels])

    jknife = JackknifePlusHDC(
        unique_labels, prototype_builder,
        sim_measure="complex_cosine", random_state=chdc.random_state,
    )

    fcp = FullConformalHDC(
        unique_labels, prototype_builder,
        sim_measure="complex_cosine", random_state=chdc.random_state,
    )
    jk_sets = jackknife_all_scores(
        jknife, enc_full, y_full, enc_test, alpha, SCORE_TYPES,
    )
    fc_sets = full_conformal_all_scores(
        fcp, enc_full, y_full, enc_test, alpha, SCORE_TYPES,
    )
    exp_results = []

    # Experiment Loop per Score Type
    for stype in SCORE_TYPES:

        chdc.compute_calib_scores(enc_cal, y_cal, score_type=stype)

        # 1. Set-Valued Prediction
        for method, marginal, model in [
            ("split_conformal", True, chdc),
            ("split_conformal", False, chdc),
            ("jackknife_plus", True, jknife),
            ("full_conformal", True, fcp),
        ]:
            if model is chdc:
                sets = model.set_valued_CP(enc_test, alpha, marginal=marginal)
            elif model is jknife:
                sets = jk_sets[stype]
            else:
                sets = fc_sets[stype]
 
            exp_results.append({
                "exp": "set_valued",
                "method": method,
                "random_state": random_state,
                "rat_id": rat_id,
                "score_type": stype,
                "alpha": alpha,
                "beta": beta,
                "marginal": marginal,
                **set_metrics(sets, y_test, unique_labels),
                # Placeholders
                "point_acc": np.nan, 
                "lc_accs":np.nan, 
                "ood_auroc": np.nan
            })
        
        # 2. Point-Valued Prediction 
        #NOTE adaptive alpha
        adap_alpha = adaptive_alpha(chdc, enc_cal, y_cal, unique_labels)
        # compute the score to choose alpha based on lc acc, then input the alpha list to this function to compute acc.
        preds_pt = chdc.point_valued_CP(enc_test, adap_alpha, allow_empty=False, marginal=False)

        # preds_pt = chdc.point_valued_CP(enc_test, alpha, allow_empty=False, marginal=False)
        # preds_pt = np.array(preds_pt).ravel()
        acc_pt = eval_accuracy(preds_pt, y_test)
        lc_accs = eval_lc_accuracy(preds_pt, y_test, unique_labels)

        exp_results.append({
            "exp": "point_valued",
            "random_state": random_state,
            "rat_id": rat_id,
            "score_type": stype,
            "alpha": alpha,
            "beta": beta,
            "point_acc": acc_pt,
            "lc_accs": lc_accs,
            # Placeholders
            "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
            "lc_covs": np.nan, "ood_auroc": np.nan
        })

        # 3. OOD Detection
        for marginal in [True, False]:
            p_vals_id  = chdc.get_max_p_value(enc_test, marginal=marginal)
            p_vals_ood = chdc.get_max_p_value(enc_ood, marginal=marginal)
            
            auroc = ood_auroc(p_vals_id, p_vals_ood)
            
            exp_results.append({
                "exp": "ood",
                "random_state": random_state,
                "rat_id": rat_id,
                "score_type": stype,
                "alpha": alpha,
                "beta": beta,
                "marginal": marginal,
                "ood_auroc": auroc,
                # Placeholders
                "set_cov": np.nan, "set_size": np.nan, "point_acc": np.nan, 
                "lc_covs": np.nan, "lc_accs": np.nan
            })

    log("Finished running conformaHDC.")

    # Baseline Vanilla HDC (Train Only)
    preds_vanilla = chdc.predict(enc_test)
    acc_vanilla = accuracy_score(y_test, preds_vanilla)
    lc_accs = eval_lc_accuracy(preds_vanilla, y_test, unique_labels)

    exp_results.append({
        "exp": "point_valued",
        "random_state": random_state,
        "rat_id": rat_id,
        "score_type": "vanilla_train",
        "alpha": alpha,
        "beta": beta,
        "point_acc": acc_vanilla,
        "lc_accs": lc_accs,
        # Placeholders
        "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
        "lc_covs": np.nan, "ood_auroc": np.nan
    })

    # Baseline Vanilla HDC (Full Train+Cal)
    vanilla_full = ConformalHDC(class_HVs=proto_full, class_labels=unique_labels, 
                                sim_measure="complex_cosine", random_state=random_state)
    preds_vanilla_full = vanilla_full.predict(enc_test)
    acc_vanilla_full = accuracy_score(y_test, preds_vanilla_full)
    lc_accs_full = eval_lc_accuracy(preds_vanilla_full, y_test, unique_labels)

    exp_results.append({
        "exp": "point_valued",
        "random_state": random_state,
        "rat_id": rat_id,
        "score_type": "vanilla_full",
        "alpha": alpha,
        "beta": beta,
        "point_acc": acc_vanilla_full,
        "lc_accs": lc_accs_full,
        # Placeholders
        "marginal": np.nan, "set_cov": np.nan, "set_size": np.nan, 
        "lc_covs": np.nan, "ood_auroc": np.nan
    })

    log("Finished running vanilla HDC.")

    return pd.DataFrame(exp_results)



# ---------------
# Main Execution
# ---------------
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python exp_rat.py <seed> <rat_id> <beta> <alpha>")
        sys.exit(1)

    seed_arg = int(sys.argv[1])
    rat_arg = int(sys.argv[2])
    alpha_arg = float(sys.argv[3])
    beta_arg = float(sys.argv[4])

    # Directory Setup
    out_dir = RESULTS_ROOT / EXP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    # rat_name = ['Barat','Buchanan','Mitt','Stella','Superchris']
    outfile = out_dir / f"rat{rat_arg}_seed{seed_arg}_alpha{alpha_arg}_beta{beta_arg}.csv"  

    print(f"Starting job: Rat {rat_arg}, Seed {seed_arg}, Alpha {alpha_arg}, Beta {beta_arg}, Reps {REPETITIONS}")

    results_list = []
    
    for i in tqdm(range(1, REPETITIONS + 1), desc="Repetitions"):
        # Unique random state per repetition
        current_state = REPETITIONS * (seed_arg - 1) + i
        
        try:
            df_rep = run_single_experiment(current_state, rat_arg, alpha_arg, beta_arg)
            results_list.append(df_rep)
        except Exception as e:
            print(f"Error in state {current_state}: {e}")

    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)
        final_df.to_csv(outfile, index=False)
        print(f"\nResults saved to {outfile}")
    else:
        print("No results generated.")
