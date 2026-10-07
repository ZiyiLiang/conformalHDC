"""
Shared common functions for the real-data conformal HDC experiments.
Each experiment script supplies its data loading and encoding policy.
"""
import os
import sys
import time
import traceback
from contextlib import contextmanager
from functools import cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from conformal_inference.fast import FastConformal, class_prototypes
from conformal_inference.methods import CachedConformalHDC, ConformalHDC
from conformal_inference.utils import eval_accuracy, eval_lc_accuracy
from data.config import RESULTS_ROOT
from hdc_encoder.level import encode_levels, make_im_cim
from hdc_encoder.image import encode_binary_images, make_position_hvs

REPETITIONS = 4
SCORE_TYPES = ["sim", "ratio", "discount", "penalized", "inverse_quantile"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Worker processes for jackknife+ and full conformal: the cores Slurm granted (1 outside Slurm)
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
# Column order of every results CSV; metrics a row does not report are NaN.
COLUMNS = ["exp", "method", "random_state", "score_type", "alpha", "marginal",
           "set_cov", "set_size", "lc_covs", "point_acc", "lc_accs", "ood_auroc",
           "macro_ap", "macro_auprc", "macro_f1"]
# Runtime table (seconds per repetition): one row per conformal method and score type.
TIMING_COLUMNS = ["random_state", "alpha", "method", "score_type", "n_test",
                  "training", "calibration", "encoding", "hdc", "conformal",
                  "device", "cpu", "n_jobs"]
TIMING = False  # set by main(timing=True); timed() is a no-op otherwise
_TIMES = {}  # stage -> wall seconds of the current repetition, filled by timed()


# Experimental utilities
def log(message):
    print(message, flush=True)


def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(stage):
    """Record the block's wall time as _TIMES[stage], including queued GPU work."""
    if not TIMING:
        yield
        return
    sync_gpu()
    start = time.perf_counter()
    yield
    sync_gpu()
    _TIMES[stage] = time.perf_counter() - start


@cache
def hardware():
    """Compute environment reported with every runtime row."""
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with open("/proc/cpuinfo") as f:
        cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    return {"device": device, "cpu": cpu, "n_jobs": N_JOBS}


def timing_frame(random_state, alpha):
    """Runtime rows of the repetition recorded in _TIMES.

    training: train + calibration encoding and prototyping; calibration: calibration
    scores (plus the adaptive-alpha search for point prediction); encoding / hdc: test
    encoding and vanilla HDC inference; conformal: overhead on top of HDC inference.
    Jackknife+ and full conformal score every type in one pass, so their rows are "all".
    """
    t = _TIMES
    base = {"random_state": random_state, "alpha": alpha, "n_test": t["n_test"],
            "training": t["encode_train"] + t["encode_cal"] + t["prototypes"],
            "encoding": t["encode_test"], "hdc": t["hdc"], **hardware()}
    rows = [{**base, "method": method, "score_type": s,
             "calibration": t[f"calib_{s}"] + t[f"adaptive_{s}"] * (method == "point_valued"),
             "conformal": t[f"{method}_{s}"]}
            for s in SCORE_TYPES for method in ["split_marginal", "split_conditional", "point_valued"]]
    rows += [{**base, "method": method, "score_type": "all", "calibration": np.nan,
              "conformal": t["fast_init"] + t[method]} for method in ["jackknife_plus", "full_conformal"]]
    return pd.DataFrame(rows, columns=TIMING_COLUMNS)


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

def masked_sims_table(chdc, hvs, alphas):
    """masked[a, i, c]: similarity of hvs[i] to class c if c is in its label-conditional
    set at alphas[a], else -inf. Scores and similarities are computed once."""
    scores, sims = chdc._test_scores(hvs), chdc._sim_matrix(hvs)
    keep = np.stack([scores <= chdc._thresholds(a, marginal=False) for a in alphas])
    return np.where(keep, sims, -np.inf), sims.argmax(axis=1)

def trimmed_predictions(masked, fallback, idx):
    """Canonical point predictions for per-class alpha indices idx."""
    per_class = masked[idx, :, np.arange(len(idx))].T  # (n, n_class)
    pred = per_class.argmax(axis=1)
    return np.where(np.isfinite(per_class).any(axis=1), pred, fallback)

def coordinate_ascent(objective, n_class, n_grid, max_passes=10):
    """Grid indices maximizing objective: best shared index first, then one class at a
    time, keeping only strict improvements so ties stay at the lower index."""
    idx = max((np.full(n_class, a) for a in range(n_grid)), key=objective)
    best = objective(idx)
    for _ in range(max_passes):
        changed = False
        for c in range(n_class):
            for a in range(n_grid):
                if a == idx[c]:
                    continue
                trial = idx.copy()
                trial[c] = a
                score = objective(trial)
                if score > best:
                    idx, best, changed = trial, score, True
        if not changed:
            break
    return idx

def adaptive_alpha(chdc, calib_hvs, calib_y, LABELS_ID, max_passes=10):
    """Per-class alpha maximizing overall calibration accuracy of the point predictor."""

    ALPHA_GRID = np.array([0.01, 0.05, 0.1, 0.15, 0.2])  # ascending: ties go to the smaller alpha

    masked, fallback = masked_sims_table(chdc, calib_hvs, ALPHA_GRID)
    y_idx = np.array([chdc.label_to_idx[l] for l in np.asarray(calib_y).tolist()])
    # evaluate the overall accuracy for each grid
    objective = lambda idx: eval_accuracy(trimmed_predictions(masked, fallback, idx), y_idx)
    return ALPHA_GRID[coordinate_ascent(objective, len(LABELS_ID), len(ALPHA_GRID), max_passes)]

################======== Metrics ========################
# Set performance metrics
def set_metrics(sets, y, labels):
    covered = np.array([label in pset for label, pset in zip(y, sets)])
    # Label conditional coverage over the classes present in y
    lc_covs = [np.mean(covered[y == lbl]) for lbl in labels if np.any(y == lbl)]
    return {
        "set_cov": np.mean(covered),
        "set_size": np.mean([len(pset) for pset in sets]),
        "lc_covs": lc_covs if lc_covs else 0.0,
    }


# Point performance metrics, score columns follow the supplied class-label order
def ranking_metrics(y, scores, labels):
    """ Equal-weight average AP and AUPRC area."""
    y, scores, labels = np.asarray(y).ravel(), np.asarray(scores), list(labels)

    aps, areas = [], []
    for c, label in enumerate(labels):
        positive = y == label
        if not positive.any() or positive.all():
            # The macro average is undefined when a class cannot be evaluated.
            return {"macro_ap": np.nan, "macro_auprc": np.nan}
        precision, recall, _ = precision_recall_curve(positive, scores[:, c])
        aps.append(average_precision_score(positive, scores[:, c]))
        areas.append(auc(recall, precision))
    return {"macro_ap": np.mean(aps), "macro_auprc": np.mean(areas)}

def point_metrics(preds, y, labels, scores):
    """Hard-prediction accuracy/F1 plus ranking metrics from continuous scores."""
    preds, y, labels = np.asarray(preds).ravel(), np.asarray(y).ravel(), list(labels)
    if not np.isin(preds, labels).all():
        raise ValueError("Predictions must belong to the supplied classes.")
    return {
        "point_acc": eval_accuracy(preds, y),
        "lc_accs": eval_lc_accuracy(preds, y, labels),
        "macro_f1": f1_score(y, preds, labels=labels, average="macro", zero_division=0),
        **ranking_metrics(y, scores, labels),
    }

# OOD performance metrics
def ood_auroc(p_id, p_ood):
    """AUROC of separating in-distribution (positive) from OOD samples by p-value."""
    y_true = np.concatenate([np.ones(len(p_id)), np.zeros(len(p_ood))])
    return roc_auc_score(y_true, np.concatenate([p_id, p_ood]))

################======== Result rows ========################
def conformal_rows(chdc, cal, test, ood_hvs, labels, jk_sets, fc_sets, random_state, alpha,
                   completion_message="Finished running ConformalHDC."):
    """Set-valued, point-valued and OOD rows for every score type.

    chdc: split-conformal model on train prototypes (similarities cached for cal/test/OOD).
    cal, test: (hvs, labels) pairs. jk_sets, fc_sets: jackknife+ and full-conformal
    prediction sets for the test HVs, keyed by score type.
    """
    (cal_hvs, cal_y), (test_hvs, test_y) = cal, test
    shared = {"random_state": random_state, "alpha": alpha}
    rows = []
    for stype in SCORE_TYPES:
        with timed(f"calib_{stype}"):
            chdc.compute_calib_scores(cal_hvs, cal_y, score_type=stype)
        shared["score_type"] = stype

        # 1. Set-Valued Prediction
        with timed(f"split_marginal_{stype}"):
            marg_sets = chdc.set_valued_CP(test_hvs, alpha, marginal=True)
        with timed(f"split_conditional_{stype}"):
            cond_sets = chdc.set_valued_CP(test_hvs, alpha, marginal=False)
        for method, marginal, sets in [
            ("split_conformal", True, marg_sets),
            ("split_conformal", False, cond_sets),
            ("jackknife_plus", True, jk_sets[stype]),
            ("full_conformal", True, fc_sets[stype]),
        ]:
            rows.append({"exp": "set_valued", "method": method, "marginal": marginal,
                         **shared, **set_metrics(sets, test_y, labels)})

        # 2. Point-Valued Prediction
        #NOTE adaptive alpha
        with timed(f"adaptive_{stype}"):
            adap_alpha = adaptive_alpha(chdc, cal_hvs, cal_y, labels)
        # compute the score to choose alpha based on lc acc, then input the alpha list to this function to compute acc.
        with timed(f"point_valued_{stype}"):
            preds = chdc.point_valued_CP(test_hvs, adap_alpha, allow_empty=False, marginal=False)
        # static alpha
        #  preds = chdc.point_valued_CP(test_hvs, alpha, allow_empty=False, marginal=False)
        rows.append({"exp": "point_valued", **shared,
                     **point_metrics(preds, test_y, labels, chdc.get_class_p_values(test_hvs))})

        # 3. OOD Detection
        for marginal in [True, False]:
            auroc = ood_auroc(chdc.get_max_p_value(test_hvs, marginal=marginal),
                              chdc.get_max_p_value(ood_hvs, marginal=marginal))
            rows.append({"exp": "ood", "marginal": marginal, **shared, "ood_auroc": auroc})
    log(completion_message)
    return rows


def vanilla_rows(models, test, random_state, alpha):
    """Evaluate vanilla models, retaining their similarity and cache policies."""
    test_hvs, test_y = test
    rows = []
    for name, model in models.items():
        rows.append({"exp": "point_valued", "random_state": random_state, "score_type": name,
                     "alpha": alpha, **point_metrics(
                         model.predict(test_hvs), test_y, model.class_labels, model._sim_matrix(test_hvs),
                     )})
    log("Finished running vanilla HDC.")
    return rows


def evaluate(train, cal, test, ood_hvs, labels, rule, random_state, alpha):
    """All result rows of one repetition from encoded (hvs, labels) splits.

    Split conformal and vanilla_train use train prototypes; jackknife+, full
    conformal and vanilla_full use train + calibration. `rule` is the prototype
    rule of conformal_inference.fast ("bipolar", "ternary" or "normalized").
    """
    _TIMES["n_test"] = len(test[1])
    with timed("prototypes"):
        protos_train = class_prototypes(*train, labels, rule)
    with timed("fast_init"):
        conformal = FastConformal(*development_set(train, cal), test[0], labels, rule,
                                  random_state=random_state, n_jobs=N_JOBS)
    log("Prototypes built.")
    chdc = CachedConformalHDC(
        class_HVs=protos_train, class_labels=labels,
        sim_measure="cosine", random_state=random_state,
    ).cache_similarities(cal[0], test[0], ood_hvs)  # re-use the similarities for every score type
    with timed("jackknife_plus"):
        jk_sets = conformal.jackknife_sets(alpha, SCORE_TYPES)
    with timed("full_conformal"):
        fc_sets = conformal.full_conformal_sets(alpha, SCORE_TYPES)
    rows = conformal_rows(chdc, cal, test, ood_hvs, labels, jk_sets, fc_sets, random_state, alpha)
    models = {
        name: ConformalHDC(prototypes, labels, sim_measure="cosine", random_state=random_state)
        for name, prototypes in {"vanilla_train": protos_train,
                                 "vanilla_full": conformal.prototypes}.items()
    }
    with timed("hdc"):  # vanilla HDC inference, similarities computed afresh
        models["vanilla_train"].predict(test[0])
    rows += vanilla_rows(models, test, random_state, alpha)
    return pd.DataFrame(rows, columns=COLUMNS)


################======== binary encoded images experiment flows ========################

def prepare_binary_images(datasets):
    """Flatten concatenated datasets and apply the original /255 pixel threshold."""
    pixels = torch.cat([d.data for d in datasets]).flatten(1)
    labels = torch.cat([d.targets for d in datasets]).numpy()
    return pixels.float().div(255) >= 0.5, labels


def balanced_indices(targets, rng):
    """Subsample each class, in sorted label order, to the smallest class size."""
    class_indices = [np.flatnonzero(targets == c) for c in np.unique(targets)]
    if not class_indices:
        raise ValueError("Image targets must be nonempty.")
    min_count = min(map(len, class_indices))
    return np.concatenate([rng.choice(idxs, min_count, replace=False) for idxs in class_indices])


def image_split_indices(targets, labels_id, labels_ood, random_state):
    """Balanced ID train/calibration/test (80/15/5%) and untrimmed OOD indices."""
    balanced = balanced_indices(targets, np.random.RandomState(random_state))
    id_idx = balanced[np.isin(targets[balanced], labels_id)]
    ood_idx = balanced[np.isin(targets[balanced], labels_ood)]
    n_total = len(id_idx)
    n_train, n_calib = int(0.8 * n_total), int(0.15 * n_total)
    splits = random_split(
        range(n_total), [n_train, n_calib, n_total - n_train - n_calib],
        generator=torch.Generator().manual_seed(random_state),
    )
    return [id_idx[split.indices] for split in splits] + [ood_idx]


def run_image_experiment(pixels, targets, labels_id, labels_ood, random_state, alpha,
                         dim=10_000, batch_size=512, replay_passes=False):
    """One binary-image repetition; replay_passes retains MNIST's loader history."""
    if pixels.ndim != 2 or len(pixels) != len(targets):
        raise ValueError("Flattened images and targets must have matching sample counts.")
    if dim <= 0 or batch_size <= 0:
        raise ValueError("Dimension and batch size must be positive.")
    seed_everything(random_state)
    splits = image_split_indices(targets, labels_id, labels_ood, random_state)
    train_idx, cal_idx, test_idx, ood_idx = splits
    pos_hvs = make_position_hvs(pixels.shape[1], dim, DEVICE)
    passes = [(len(train_idx), True)]
    if replay_passes:
        # Preserve MNIST's original RNG draws before its final training pass.
        passes += [(len(train_idx) + len(cal_idx), True), (len(cal_idx), False),
                   (len(test_idx), False), (len(ood_idx), False), (len(train_idx), True)]
    train_order = loader_orders(passes, batch_size)[-1]

    def encode(indices):
        return encode_binary_images(pixels[indices], pos_hvs, batch_size), targets[indices]

    with timed("encode_train"):
        train = encode(train_idx[train_order])
    with timed("encode_cal"):
        cal = encode(cal_idx)
    with timed("encode_test"):
        test = encode(test_idx)
    ood_hvs, _ = encode(balance_ood(ood_idx, len(test_idx)))
    log("Encoding complete.")
    return evaluate(train, cal, test, ood_hvs, labels_id, "bipolar", random_state, alpha)


################======== Image experiment flows ========################

# Level-encoded experiment flow

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
    with timed("encode_train"):
        train = encode_levels(X_train, iM, CiM), y_train
    with timed("encode_cal"):
        cal = encode_levels(X_cal, iM, CiM), y_cal
    with timed("encode_test"):
        test = encode_levels(X_test, iM, CiM), y_test
    ood_hvs = encode_levels(balance_ood(L[np.isin(y, labels_ood)], len(X_test)), iM, CiM)
    log("Encoding complete.")
    return evaluate(train, cal, test, ood_hvs, labels_id, "ternary", random_state, alpha)


################======== Main Execution ========################

def main(run_single_experiment, exp_name, timing=False):
    """Run REPETITIONS seeds of one seed group and save them to RESULTS_ROOT/<exp_name>/."""
    if len(sys.argv) != 3:
        log(f"Usage: python {Path(sys.argv[0]).name} <seed_group_id> <alpha>")
        sys.exit(1)
    seed_arg, alpha_arg = int(sys.argv[1]), float(sys.argv[2])
    out_dir = RESULTS_ROOT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"seed{seed_arg}_alpha{alpha_arg}.csv"
    log(f"Starting job: Seed Group {seed_arg}, Alpha {alpha_arg}, Reps {REPETITIONS}")

    global TIMING
    TIMING = timing  # also write timing_<outfile> with the runtime breakdown
    results, timings = [], []
    for i in tqdm(range(1, REPETITIONS + 1), desc="Repetitions"):
        state = REPETITIONS * (seed_arg - 1) + i
        log(f"Running repetition {i}...")
        _TIMES.clear()
        try:
            results.append(run_single_experiment(state, alpha_arg))
            if timing:
                timings.append(timing_frame(state, alpha_arg))
        except Exception:
            log(f"Error in state {state}:")
            traceback.print_exc()
            sys.stderr.flush()

    if results:
        pd.concat(results, ignore_index=True).to_csv(outfile, index=False)
        log(f"\nResults saved to {outfile}")
    if timings:
        timefile = out_dir / f"timing_{outfile.name}"
        pd.concat(timings, ignore_index=True).to_csv(timefile, index=False)
        log(f"Timings saved to {timefile}")
    else:
        log("No results generated.")
