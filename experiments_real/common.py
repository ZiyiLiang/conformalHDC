"""Shared harness for the real-data conformal HDC experiments.

Each experiment script supplies its data loading and encoding plus a
``run_single_experiment(random_state, alpha)`` that returns ``results_frame(rows)``;
this module turns encoded hypervectors into result rows and runs the repetitions.
"""
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from conformal_inference.fast import FastConformal, class_prototypes
from conformal_inference.methods import CachedConformalHDC, ConformalHDC
from conformal_inference.utils import eval_accuracy, eval_lc_accuracy
from hdc_encoder.level import encode_levels, make_im_cim

REPETITIONS = 4 
SCORE_TYPES = ["sim", "ratio", "discount", "penalized", "inverse_quantile"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Worker processes for jackknife+ and full conformal: the cores Slurm granted (1 outside Slurm)
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
# Column order of every results CSV; metrics a row does not report are NaN.
COLUMNS = ["exp", "method", "random_state", "score_type", "alpha", "marginal",
           "set_cov", "set_size", "lc_covs", "point_acc", "lc_accs", "ood_auroc"]


def log(message):
    print(message, flush=True)


def seed_everything(random_state):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def loader_orders(passes, batch_size):
    """Sample visit order of consecutive DataLoader passes, given as (n_samples, shuffle).

    Every pass draws from torch's global RNG, and a shuffled pass also draws its
    permutation. Replaying the passes over index tensors reproduces the original
    sample order without loading or encoding the data again.
    """
    return [
        torch.cat([torch.arange(0)] + list(
            DataLoader(torch.arange(n), batch_size=batch_size, shuffle=shuffle)
        )).numpy()
        for n, shuffle in passes
    ]


def development_set(train, cal):
    """Train + calibration (hvs, labels), the data jackknife+ and full conformal fit on."""
    return np.concatenate((train[0], cal[0])), np.concatenate((train[1], cal[1]))


def balance_ood(ood, n_test):
    """Truncate the OOD split to at most as many samples as the test split."""
    return ood[:min(len(ood), n_test)]


################======== Metrics ========################

def set_metrics(sets, y, labels):
    covered = np.array([label in pset for label, pset in zip(y, sets)])
    # Label conditional coverage over the classes present in y
    lc_covs = [np.mean(covered[y == lbl]) for lbl in labels if np.any(y == lbl)]
    return {
        "set_cov": np.mean(covered),
        "set_size": np.mean([len(pset) for pset in sets]),
        "lc_covs": lc_covs if lc_covs else 0.0,
    }


def point_metrics(preds, y, labels):
    preds = np.array(preds).ravel()
    return {"point_acc": eval_accuracy(preds, y),
            "lc_accs": eval_lc_accuracy(preds, y, labels)}


def ood_auroc(p_id, p_ood):
    """AUROC of separating in-distribution (positive) from OOD samples by p-value."""
    y_true = np.concatenate([np.ones(len(p_id)), np.zeros(len(p_ood))])
    return roc_auc_score(y_true, np.concatenate([p_id, p_ood]))


################======== Result rows ========################

def conformal_rows(chdc, cal, test, ood_hvs, labels, jk_sets, fc_sets, random_state, alpha):
    """Set-valued, point-valued and OOD rows for every score type.

    chdc: split-conformal model on train prototypes (similarities cached for cal/test/OOD).
    cal, test: (hvs, labels) pairs. jk_sets, fc_sets: jackknife+ and full-conformal
    prediction sets for the test HVs, keyed by score type.
    """
    (cal_hvs, cal_y), (test_hvs, test_y) = cal, test
    shared = {"random_state": random_state, "alpha": alpha}
    rows = []
    for stype in SCORE_TYPES:
        chdc.compute_calib_scores(cal_hvs, cal_y, score_type=stype)
        shared["score_type"] = stype

        # 1. Set-Valued Prediction
        for method, marginal, sets in [
            ("split_conformal", True, chdc.set_valued_CP(test_hvs, alpha, marginal=True)),
            ("split_conformal", False, chdc.set_valued_CP(test_hvs, alpha, marginal=False)),
            ("jackknife_plus", True, jk_sets[stype]),
            ("full_conformal", True, fc_sets[stype]),
        ]:
            rows.append({"exp": "set_valued", "method": method, "marginal": marginal,
                         **shared, **set_metrics(sets, test_y, labels)})

        # 2. Point-Valued Prediction
        preds = chdc.point_valued_CP(test_hvs, alpha, allow_empty=False, marginal=False)
        rows.append({"exp": "point_valued", **shared, **point_metrics(preds, test_y, labels)})

        # 3. OOD Detection
        for marginal in [True, False]:
            auroc = ood_auroc(chdc.get_max_p_value(test_hvs, marginal=marginal),
                              chdc.get_max_p_value(ood_hvs, marginal=marginal))
            rows.append({"exp": "ood", "marginal": marginal, **shared, "ood_auroc": auroc})
    log("Finished running ConformalHDC.")
    return rows


def vanilla_rows(prototypes, labels, test, random_state, alpha):
    """Point-valued rows of plain nearest-prototype HDC, one per {score_type: prototypes}."""
    test_hvs, test_y = test
    rows = []
    for name, class_HVs in prototypes.items():
        model = ConformalHDC(class_HVs, labels, sim_measure="cosine", random_state=random_state)
        rows.append({"exp": "point_valued", "random_state": random_state, "score_type": name,
                     "alpha": alpha, **point_metrics(model.predict(test_hvs), test_y, labels)})
    log("Finished running vanilla HDC.")
    return rows


def results_frame(rows, columns=COLUMNS):
    return pd.DataFrame(rows, columns=columns)


def evaluate(train, cal, test, ood_hvs, labels, rule, random_state, alpha):
    """All result rows of one repetition from encoded (hvs, labels) splits.

    Split conformal and vanilla_train use train prototypes; jackknife+, full
    conformal and vanilla_full use train + calibration. `rule` is the prototype
    rule of conformal_inference.fast ("bipolar", "ternary" or "normalized").
    """
    protos_train = class_prototypes(*train, labels, rule)
    conformal = FastConformal(*development_set(train, cal), test[0], labels, rule,
                              random_state=random_state, n_jobs=N_JOBS)
    log("Prototypes built.")
    chdc = CachedConformalHDC(
        class_HVs=protos_train, class_labels=labels,
        sim_measure="cosine", random_state=random_state,
    ).cache_similarities(cal[0], test[0], ood_hvs)  # re-use the similarities for every score type
    rows = conformal_rows(
        chdc, cal, test, ood_hvs, labels,
        conformal.jackknife_sets(alpha, SCORE_TYPES),
        conformal.full_conformal_sets(alpha, SCORE_TYPES),
        random_state, alpha,
    )
    rows += vanilla_rows(
        {"vanilla_train": protos_train, "vanilla_full": conformal.prototypes},
        labels, test, random_state, alpha,
    )
    return results_frame(rows)


################======== Shared experiment flows ========################

def run_level_experiment(L, y, labels_id, labels_ood, test_size, levels, random_state, alpha,
                         dim=10_000):
    """One repetition on quantized tabular features L (UCI HAR, ISOLET): split the ID
    data into train/calib/test, level-encode, and evaluate with ternary prototypes."""
    seed_everything(random_state)
    id_mask = np.isin(y, labels_id)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        L[id_mask], y[id_mask], test_size=test_size, random_state=random_state, stratify=y[id_mask],
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.4, random_state=random_state, stratify=y_train_full,
    )
    log(f"ID: {len(X_train)} train, {len(X_cal)} calib, {len(X_test)} test.")

    iM, CiM = make_im_cim(L.shape[1], levels, dim, DEVICE)
    train = encode_levels(X_train, iM, CiM), y_train
    cal = encode_levels(X_cal, iM, CiM), y_cal
    test = encode_levels(X_test, iM, CiM), y_test
    ood_hvs = encode_levels(balance_ood(L[np.isin(y, labels_ood)], len(X_test)), iM, CiM)
    log("Encoding complete.")
    return evaluate(train, cal, test, ood_hvs, labels_id, "ternary", random_state, alpha)


################======== Main Execution ========################

def main(run_single_experiment, exp_name):
    """Run REPETITIONS seeds of one seed group and save them to results/<exp_name>/."""
    if len(sys.argv) != 3:
        log(f"Usage: python {Path(sys.argv[0]).name} <seed_group_id> <alpha>")
        sys.exit(1)
    seed_arg, alpha_arg = int(sys.argv[1]), float(sys.argv[2])
    out_dir = Path(f"./results/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"seed{seed_arg}_alpha{alpha_arg}.csv"
    log(f"Starting job: Seed Group {seed_arg}, Alpha {alpha_arg}, Reps {REPETITIONS}")

    results = []
    for i in tqdm(range(1, REPETITIONS + 1), desc="Repetitions"):
        state = REPETITIONS * (seed_arg - 1) + i
        log(f"Running repetition {i}...")
        try:
            results.append(run_single_experiment(state, alpha_arg))
        except Exception:
            log(f"Error in state {state}:")
            traceback.print_exc()
            sys.stderr.flush()

    if results:
        pd.concat(results, ignore_index=True).to_csv(outfile, index=False)
        log(f"\nResults saved to {outfile}")
    else:
        log("No results generated.")
