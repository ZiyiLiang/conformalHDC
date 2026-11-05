"""Shared OOD experiments: matched HDC representations, ID-only cutoffs and raw AUROC.

Run the dataset entrypoints, not this module. The four main-paper datasets use
their existing encoders and splits; tabular quantization is fitted on ID training
features only. Every declared OOD observation is evaluated, without prefix cropping.
"""
import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from conformal_inference.fast import class_prototypes
from data.config import DATA_ROOT
from hdc_encoder.image import encode_binary_images, make_position_hvs
from hdc_encoder.level import encode_levels, make_im_cim

SCHEMA_VERSION = 1
METHODS = ("maxsim", "discount", "energy")
ALPHA_MULTIPLIERS = np.array([0.5, 0.75, 1.0, 1.25, 1.5])


@dataclass(frozen=True)
class Experiment:
    labels_id: tuple
    labels_ood: tuple
    nominal_alpha: float
    encoding: str


EXPERIMENTS = {
    "mnist": Experiment((0, 1, 2, 3, 4, 5), (6, 7, 8, 9), 0.05, "image"),
    "fashion_mnist": Experiment((0, 1, 2, 3, 4, 6), (5, 7, 8, 9), 0.05, "image"),
    "pamap2": Experiment((0, 2), (1,), 0.1, "level"),
    "har": Experiment((0, 1, 2), (3, 4, 5), 0.1, "level"),
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def source_hashes(dataset):
    files = [Path(__file__), Path(__file__).with_name(f"ood_{dataset}.py"),
             PROJECT_ROOT / "data/load.py", PROJECT_ROOT / "hdc_encoder/image.py",
             PROJECT_ROOT / "hdc_encoder/level.py", PROJECT_ROOT / "conformal_inference/fast.py",
             PROJECT_ROOT / "experiments_real/common.py"]
    if dataset == "pamap2":
        files.append(PROJECT_ROOT / "experiments_real/exp_pamap2.py")
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files}


def load_inputs(dataset, data_root):
    """Read cached datasets only. No network downloads or fabricated substitutes."""
    spec = EXPERIMENTS[dataset]
    if spec.encoding == "image":
        from torchvision import datasets
        from experiments_real.common import prepare_binary_images
        factory = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        parts = [factory(str(data_root), train=train, download=False) for train in (True, False)]
        pixels, y = prepare_binary_images(parts)
        return pixels, y
    # The existing data loaders bind DATA_ROOT at import time. Override it only
    # inside this experiment process; the shared repository configuration is untouched.
    from data import load as data_load
    data_load.DATA_ROOT = Path(data_root)
    if dataset == "har":
        return data_load.load_har_data()
    X, activities = data_load.load_pamap2_data("Protocol")
    from experiments_real.exp_pamap2 import INTENSITY_ACTIVITIES
    y = np.full(len(activities), -1, dtype=np.int64)
    for intensity, activity_ids in enumerate(INTENSITY_ACTIVITIES):
        y[np.isin(activities, activity_ids)] = intensity
    if np.any(y < 0):
        raise ValueError("PAMAP2 contains activities outside the main experiment's intensity map")
    return X, y


def split_indices(y, spec, random_state):
    """Main-paper ID splits, with every intended OOD example retained."""
    if spec.encoding == "image":
        from experiments_real.common import image_split_indices
        parts = image_split_indices(y, spec.labels_id, spec.labels_ood, random_state)
        parts[3] = np.flatnonzero(np.isin(y, spec.labels_ood))
    else:
        id_indices = np.flatnonzero(np.isin(y, spec.labels_id))
        development, test = train_test_split(
            id_indices, test_size=0.1, random_state=random_state, stratify=y[id_indices])
        train, calibration = train_test_split(
            development, test_size=0.4, random_state=random_state, stratify=y[development])
        parts = [train, calibration, test, np.flatnonzero(np.isin(y, spec.labels_ood))]
    for indices in parts:
        if not len(indices):
            raise ValueError("Every training, calibration, ID-test and OOD partition must be nonempty")
    observed_ood = set(np.unique(y[parts[3]]).tolist())
    if observed_ood != set(spec.labels_ood):
        raise ValueError(f"Missing intended OOD classes: expected {spec.labels_ood}, found {observed_ood}")
    return parts


def fit_quantizer(X_train):
    X_train = np.asarray(X_train)
    if not np.isfinite(X_train).all():
        raise ValueError("Training features must be finite")
    minimum = X_train.min(axis=0)
    span = np.maximum(X_train.max(axis=0) - minimum, 1e-8)
    return minimum, span


def apply_quantizer(X, minimum, span, levels=21):
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite")
    return np.clip(np.round((X - minimum) / span * (levels - 1)), 0, levels - 1).astype(np.int64)


def encode_partitions(X, y, indices, spec, dimension, device):
    train_indices = indices[0]
    metadata = {}
    if spec.encoding == "image":
        position_hvs = make_position_hvs(X.shape[1], dimension, device)
        encoded = [encode_binary_images(X[idx], position_hvs) for idx in indices]
        rule = "bipolar"
    else:
        minimum, span = fit_quantizer(X[train_indices])
        metadata["quantizer_sha256"] = array_digest(minimum, span)
        metadata["quantizer_fit"] = "ID training features only; held-out features clipped to training ranges"
        item_memory, level_memory = make_im_cim(X.shape[1], 21, dimension, device)
        encoded = [encode_levels(apply_quantizer(X[idx], minimum, span), item_memory, level_memory)
                   for idx in indices]
        rule = "ternary"
    prototypes = class_prototypes(encoded[0], y[train_indices], spec.labels_id, rule)
    return encoded, prototypes, metadata


def cosine_similarities(H, prototypes):
    H, prototypes = np.asarray(H), np.asarray(prototypes)
    norms = np.linalg.norm(H, axis=1)[:, None] * np.linalg.norm(prototypes, axis=1)[None, :]
    cosine = np.divide(H @ prototypes.T, norms,
                       out=np.zeros((len(H), len(prototypes)), dtype=np.float32), where=norms > 0)
    return np.clip(cosine, -1.0, 1.0)


def ood_scores(cosine, temperature=1.0):
    """All scores are oriented so larger values mean ID; no calibration is used."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("Energy temperature must be positive and finite")
    cosine = np.asarray(cosine, dtype=float)
    if cosine.ndim != 2 or not np.isfinite(cosine).all():
        raise ValueError("Expected a finite sample-by-class cosine matrix")
    delta = (1.0 + cosine) / 2.0
    total = delta.sum(axis=1)
    discount = np.divide(delta.max(axis=1) ** 2, total, out=np.zeros(len(delta)), where=total > 0)
    return {"maxsim": cosine.max(axis=1), "discount": discount,
            "energy": temperature * logsumexp(cosine / temperature, axis=1)}


def lower_tail_threshold(calibration_scores, alpha):
    """Reject score < threshold; ties are retained and finite-sample cutoffs conservative.

    With n ID-calibration scores, k=floor(alpha*(n+1)); use the kth order
    statistic (1-based), or -inf for k=0. No OOD data select the threshold.
    """
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a nonempty finite vector")
    if not 0 < alpha < 1:
        raise ValueError("Operating alpha must lie in (0,1)")
    k = min(len(scores), int(np.floor(alpha * (len(scores) + 1) + 1e-12)))
    return -np.inf if k == 0 else float(np.partition(scores, k - 1)[k - 1])


def class_counts(y):
    labels, counts = np.unique(y, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels.tolist(), counts)}


def run_repetition(dataset, X, y, random_state, args):
    spec = EXPERIMENTS[dataset]
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    indices = split_indices(y, spec, random_state)
    encoded, prototypes, encoding_metadata = encode_partitions(
        X, y, indices, spec, args.dimension, torch.device(args.device))
    scores = [ood_scores(cosine_similarities(H, prototypes), args.temperature) for H in encoded[1:]]
    calibration, test, ood = scores
    sizes = dict(zip(("n_train", "n_cal", "n_test", "n_ood"), map(len, indices)))
    shared = {"dataset": dataset, "random_state": random_state, "nominal_alpha": spec.nominal_alpha,
              "temperature": args.temperature, "dimension": args.dimension, **sizes}
    auc_rows, operating_rows = [], []
    true_id = np.concatenate([np.ones(sizes["n_test"]), np.zeros(sizes["n_ood"])])
    for method in METHODS:
        auc_rows.append({**shared, "method": method,
                         "ood_auroc": roc_auc_score(true_id, np.concatenate([test[method], ood[method]]))})
        for alpha in args.alpha_grid:
            threshold = lower_tail_threshold(calibration[method], alpha)
            id_rejected = int(np.count_nonzero(test[method] < threshold))
            ood_rejected = int(np.count_nonzero(ood[method] < threshold))
            operating_rows.append({**shared, "method": method, "alpha": alpha, "threshold": threshold,
                                   "id_rejected": id_rejected, "ood_rejected": ood_rejected,
                                   "id_false_rejection": id_rejected / sizes["n_test"],
                                   "ood_rejection": ood_rejected / sizes["n_ood"]})
    metadata = {"random_state": random_state, **sizes, **encoding_metadata,
                "class_counts": {name: class_counts(y[idx]) for name, idx in zip(
                    ("train", "calibration", "test", "ood"), indices)},
                "split_indices_sha256": array_digest(*indices)}
    return auc_rows, operating_rows, metadata


def parser(dataset):
    result = argparse.ArgumentParser(description=f"OOD score comparison for {dataset}")
    result.add_argument("seed_group", type=int)
    result.add_argument("--results-root", type=Path,
                        default=Path(os.environ.get("CHDC_RESULTS_ROOT", PROJECT_ROOT / "results")))
    result.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("CHDC_DATA_ROOT", DATA_ROOT)))
    result.add_argument("--repetitions", type=int, default=4)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--dimension", type=int, default=10_000)
    result.add_argument("--alpha-grid", nargs="+", type=float)
    result.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    result.add_argument("--force", action="store_true")
    result.add_argument("--check-complete", action="store_true",
                        help="Validate matching completion marker and CSVs without loading datasets")
    return result


def run_config(dataset, args):
    return {"schema_version": SCHEMA_VERSION, "dataset": dataset, "seed_group": args.seed_group,
            "repetitions": args.repetitions, "random_states": [
                args.repetitions * (args.seed_group - 1) + i for i in range(1, args.repetitions + 1)],
            "methods": list(METHODS), "alpha_grid": args.alpha_grid,
            "nominal_alpha": EXPERIMENTS[dataset].nominal_alpha, "temperature": args.temperature,
            "dimension": args.dimension, "device": args.device, "threads": args.threads,
            "data_root": str(args.data_root.resolve()), "source_hashes": source_hashes(dataset),
            "package_versions": {name: version(name) for name in
                                 ("numpy", "pandas", "scipy", "scikit-learn", "torch")}}


def is_complete(marker_path, config):
    try:
        marker = json.loads(marker_path.read_text())
        if marker["run_config"] != config:
            return False
        for key, multiplier in (("auroc", 1), ("operating", len(config["alpha_grid"]))):
            path = marker_path.parent / marker["files"][key]
            if sha256_file(path) != marker["sha256"][key]:
                return False
            frame = pd.read_csv(path)
            if len(frame) != config["repetitions"] * len(METHODS) * multiplier:
                return False
            if set(frame["random_state"]) != set(config["random_states"]) or set(frame["method"]) != set(METHODS):
                return False
            columns = ["random_state", "method"] + (["alpha"] if key == "operating" else [])
            if frame.duplicated(columns).any():
                return False
            if key == "operating" and not np.allclose(sorted(frame["alpha"].unique()), config["alpha_grid"]):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(dataset):
    args = parser(dataset).parse_args()
    if min(args.seed_group, args.repetitions, args.dimension, args.threads) < 1:
        raise ValueError("Seed group, repetitions, dimension and threads must be positive")
    if args.temperature <= 0 or not np.isfinite(args.temperature):
        raise ValueError("Energy temperature must be positive and finite")
    args.alpha_grid = sorted(set(round(float(a), 12) for a in (args.alpha_grid or (
        EXPERIMENTS[dataset].nominal_alpha * ALPHA_MULTIPLIERS).tolist())))
    if not args.alpha_grid or any(not np.isfinite(a) or not 0 < a < 1 for a in args.alpha_grid):
        raise ValueError("Alpha grid must contain finite values in (0,1)")
    config = run_config(dataset, args)
    output = args.results_root / f"ood_{dataset}"
    prefix = f"seed{args.seed_group}"
    marker_path = output / f"{prefix}_complete.json"
    complete = is_complete(marker_path, config)
    if args.check_complete:
        raise SystemExit(0 if complete else 1)
    if complete and not args.force:
        print(f"Already complete: {marker_path}", flush=True)
        return
    marker_path.unlink(missing_ok=True)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(f"Dataset={dataset}; seed group={args.seed_group}; reps={args.repetitions}; "
          f"data={args.data_root}; results={output}; device={args.device}", flush=True)
    X, y = load_inputs(dataset, args.data_root)
    input_hash = array_digest(X.numpy() if isinstance(X, torch.Tensor) else X, y)
    auc_rows, operating_rows, repetitions = [], [], []
    for state in config["random_states"]:
        print(f"Starting repetition {state}", flush=True)
        with threadpool_limits(limits=args.threads):
            auc, operating, metadata = run_repetition(dataset, X, y, state, args)
        auc_rows.extend(auc)
        operating_rows.extend(operating)
        repetitions.append(metadata)
        print(f"Finished repetition {state}: ID test={metadata['n_test']}, OOD={metadata['n_ood']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous marker before replacing files. A failed job cannot
    # leave a partially updated result pair marked complete.
    marker_path.unlink(missing_ok=True)
    files = {"auroc": f"{prefix}_ood.csv", "operating": f"{prefix}_operating.csv"}
    for name, rows in (("auroc", auc_rows), ("operating", operating_rows)):
        target = output / files[name]
        temporary = target.with_suffix(target.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(target)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    marker = {**config, "run_config": config, "files": files,
              "sha256": {key: sha256_file(output / name) for key, name in files.items()},
              "input_array_sha256": input_hash, "repetition_metadata": repetitions,
              "git_revision": revision, "score_direction": "larger means ID",
              "energy_input": "raw cosine in [-1,1]", "threshold_rule": "score < threshold",
              "calibration_rule": "k=floor(alpha*(n_cal+1)); kth ascending ID-calibration score; -inf when k=0",
              "packages": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}}
    temporary = marker_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(marker_path)
    if not is_complete(marker_path, config):
        marker_path.unlink(missing_ok=True)
        raise RuntimeError("Output validation failed; no completion marker retained")
    print(f"Completed: {marker_path}", flush=True)
