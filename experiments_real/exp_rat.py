"""Rat odor-decoding experiments using complex hypervectors and refit conformal."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from hdc_encoder.neuro import RFF
from conformal_inference.methods import CachedConformalHDC, ConformalHDC
from conformal_inference.refit import (
    JackknifePlusHDC, FullConformalHDC, jackknife_all_scores, full_conformal_all_scores,
)
from data.load import load_rat_data
from data.config import RESULTS_ROOT
from experiments_real.common import (
    REPETITIONS, SCORE_TYPES, conformal_rows, development_set, log, vanilla_rows,
)

EXP_NAME = "odor_decoding"
DIM = 15_000
TRAINING_WINDOW = (200, 600)
RUNNING_WINDOW = (0, 200)
BIN_SIZE = 25
SLICING_WINDOW = 200
STEP_BINS = 2
# Preserve the original RAT schema, including the positions of the point metrics.
COLUMNS = ["exp", "method", "random_state", "rat_id", "score_type", "alpha", "beta",
           "marginal", "set_cov", "set_size", "lc_covs", "point_acc", "lc_accs",
           "ood_auroc", "macro_ap", "macro_auprc", "macro_f1"]


def load_splits(random_state, rat_id):
    splits, ood = load_rat_data(
        irat=rat_id, split_ratio=(0.5, 0.4, 0.1), training_window=TRAINING_WINDOW,
        running_window=RUNNING_WINDOW, step_bins=STEP_BINS,
        slicing_window=SLICING_WINDOW, bin_size=BIN_SIZE, seed=random_state,
    )
    log("Data loaded.")
    return [(part.X, part.y) for part in (splits.train, splits.cal, splits.test)], ood


def encode_splits(splits, ood, random_state, beta):
    rff = RFF(n_feature=splits[0][0].shape[2], dimension=DIM, seed=random_state)
    basis = rff.gen_basis(cov=np.eye(rff.n_feature))
    time_base = rff.gen_time_base()
    encoded = [(rff.encode_all(basis, X, time_base, beta=beta), y) for X, y in splits]
    return rff, encoded, rff.encode_all(basis, ood, time_base, beta=beta)


def build_prototypes(rff, hvs, y, labels):
    """Ordered complex prototypes; refit subsets missing a class use zeros."""
    prototypes = rff.build_class_prototypes(hvs, y)
    empty = np.zeros(hvs.shape[1], dtype=hvs.dtype)
    return np.stack([prototypes.get(label, empty) for label in labels])


def prepare_models(rff, train, cal, test, ood_hvs, labels, random_state):
    full = development_set(train, cal)
    proto_train = build_prototypes(rff, *train, labels)
    proto_full = build_prototypes(rff, *full, labels)
    log("Prototypes built.")
    # Split conformal historically uses seed 0; retain it for randomized scores.
    chdc = CachedConformalHDC(proto_train, labels, sim_measure="complex_cosine").cache_similarities(
        cal[0], test[0], ood_hvs,
    )
    vanilla_full = ConformalHDC(proto_full, labels, sim_measure="complex_cosine",
                                random_state=random_state)
    return chdc, vanilla_full, full


def refit_sets(rff, full, test_hvs, labels, alpha, random_state):
    def prototype_builder(hvs, y, class_labels):
        return build_prototypes(rff, hvs, y, class_labels)

    jknife = JackknifePlusHDC(labels, prototype_builder, sim_measure="complex_cosine",
                              random_state=random_state)
    fcp = FullConformalHDC(labels, prototype_builder, sim_measure="complex_cosine",
                           random_state=random_state)
    jk_sets = jackknife_all_scores(jknife, *full, test_hvs, alpha, SCORE_TYPES)
    fc_sets = full_conformal_all_scores(fcp, *full, test_hvs, alpha, SCORE_TYPES)
    return jk_sets, fc_sets


def evaluate_rat(rff, train, cal, test, ood_hvs, random_state, rat_id, alpha, beta):
    labels = sorted(np.unique(train[1]))
    chdc, vanilla_full, full = prepare_models(rff, train, cal, test, ood_hvs, labels, random_state)
    jk_sets, fc_sets = refit_sets(rff, full, test[0], labels, alpha, chdc.random_state)
    rows = conformal_rows(
        chdc, cal, test, ood_hvs, labels, jk_sets, fc_sets, random_state, alpha,
        completion_message="Finished running conformaHDC.",
    )
    rows += vanilla_rows({"vanilla_train": chdc, "vanilla_full": vanilla_full},
                               test, random_state, alpha)
    return pd.DataFrame([{**row, "rat_id": rat_id, "beta": beta} for row in rows], columns=COLUMNS)


def run_single_experiment(random_state, rat_id, alpha, beta):
    splits, ood = load_splits(random_state, rat_id)
    rff, (train, cal, test), ood_hvs = encode_splits(splits, ood, random_state, beta)
    return evaluate_rat(rff, train, cal, test, ood_hvs, random_state, rat_id, alpha, beta)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python exp_rat.py <seed> <rat_id> <alpha> <beta>")
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
