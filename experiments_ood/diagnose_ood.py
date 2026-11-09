"""Export OOD ROC curves and observation scores without replacing legacy results.

All scientific operations use experiments_ood.common. This runner only adds
diagnostic exports under results/ood_diagnostics; legacy result comparisons live
outside the repository in the temporary verification workflow.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from experiments_ood import common

DIAGNOSTIC_SCHEMA_VERSION = 1
PARTITIONS = ("calibration", "test", "ood")
SCORE_COLUMNS = ("dataset", "random_state", "partition", "sample_index", "label", *common.METHODS)
ROC_COLUMNS = ("dataset", "random_state", "method", "id_false_rejection", "ood_rejection", "detection_threshold")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("seed_group", type=int)
    result.add_argument("--dataset", choices=tuple(common.EXPERIMENTS), required=True)
    result.add_argument("--results-root", type=Path,
                        default=Path(os.environ.get("CHDC_RESULTS_ROOT", PROJECT_ROOT / "results")))
    result.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("CHDC_DATA_ROOT", PROJECT_ROOT / "data")))
    result.add_argument("--repetitions", type=int, default=5)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--dimension", type=int, default=10_000)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    result.add_argument("--alpha-grid", nargs="+", type=float)
    result.add_argument("--check-complete", action="store_true",
                        help="Validate a matching diagnostic export without loading the dataset")
    return result


def run_config(dataset, args):
    config = common.run_config(dataset, args)
    config["diagnostic_schema_version"] = DIAGNOSTIC_SCHEMA_VERSION
    config["source_hashes"] = {
        **config["source_hashes"],
        "experiments_ood/diagnose_ood.py": common.sha256_file(__file__),
    }
    return config


def file_names(group):
    prefix = f"seed{group}"
    return {"scores": f"{prefix}_scores.csv.gz", "roc": f"{prefix}_roc.csv.gz",
            "auroc": f"{prefix}_ood.csv", "operating": f"{prefix}_operating.csv"}


def output_folder(args):
    namespace = args.results_root.resolve() / "ood_diagnostics"
    output = namespace / f"ood_{args.dataset}"
    # A redirected namespace must not silently route exports into legacy folders.
    if namespace.resolve() != namespace or output.resolve() != output:
        raise ValueError("The diagnostic namespace and dataset directory must not be symbolic links")
    return output


def export_repetition(dataset, X, y, random_state, args):
    """Use the same seeds, partitions, encoding and metric order as the legacy run."""
    spec = common.EXPERIMENTS[dataset]
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    indices = common.split_indices(y, spec, random_state)
    encoded, prototypes, encoding_metadata = common.encode_partitions(
        X, y, indices, spec, args.dimension, torch.device(args.device))
    scores = [common.ood_scores(common.cosine_similarities(H, prototypes), args.temperature)
              for H in encoded[1:]]
    calibration, test, ood = scores
    sizes = dict(zip(("n_train", "n_cal", "n_test", "n_ood"), map(len, indices)))
    shared = {"dataset": dataset, "random_state": random_state, "nominal_alpha": spec.nominal_alpha,
              "temperature": args.temperature, "dimension": args.dimension, **sizes}

    score_frames = [pd.DataFrame({"dataset": dataset, "random_state": random_state,
                                 "partition": partition, "sample_index": idx, "label": y[idx], **values})
                    for partition, idx, values in zip(PARTITIONS, indices[1:], scores)]
    true_id = np.concatenate([np.ones(sizes["n_test"]), np.zeros(sizes["n_ood"])])
    auc_rows, operating_rows, roc_frames = [], [], []
    for method in common.METHODS:
        id_score = np.concatenate([test[method], ood[method]])
        auc_rows.append({**shared, "method": method,
                         "ood_auroc": common.roc_auc_score(true_id, id_score)})
        fpr, tpr, threshold = roc_curve(1 - true_id, -id_score, drop_intermediate=False)
        roc_frames.append(pd.DataFrame({"dataset": dataset, "random_state": random_state,
                                        "method": method, "id_false_rejection": fpr,
                                        "ood_rejection": tpr, "detection_threshold": threshold}))
        for alpha in args.alpha_grid:
            threshold = common.lower_tail_threshold(calibration[method], alpha)
            id_rejected = int(np.count_nonzero(test[method] < threshold))
            ood_rejected = int(np.count_nonzero(ood[method] < threshold))
            operating_rows.append({**shared, "method": method, "alpha": alpha, "threshold": threshold,
                                   "id_rejected": id_rejected, "ood_rejected": ood_rejected,
                                   "id_false_rejection": id_rejected / sizes["n_test"],
                                   "ood_rejection": ood_rejected / sizes["n_ood"]})
    metadata = {"random_state": random_state, **sizes, **encoding_metadata,
                "class_counts": {name: common.class_counts(y[idx]) for name, idx in zip(
                    ("train", "calibration", "test", "ood"), indices)},
                "split_indices_sha256": common.array_digest(*indices)}
    return {"scores": pd.concat(score_frames, ignore_index=True),
            "roc": pd.concat(roc_frames, ignore_index=True),
            "auroc": pd.DataFrame(auc_rows), "operating": pd.DataFrame(operating_rows)}, metadata


def validate_export(marker, output, config):
    """Validate this export's schema and internal integrity, without reading legacy results."""
    if (not common.run_configs_match(marker.get("run_config"), config, marker.get("git_revision")) or
            any(marker.get(key) != value for key, value in marker["run_config"].items())):
        raise ValueError("Diagnostic configuration does not match the requested run")
    if marker.get("files") != file_names(config["seed_group"]):
        raise ValueError("Unexpected diagnostic filenames")
    if marker.get("roc_positive_class") != "ood" or marker.get("threshold_rule") != "id_score < threshold":
        raise ValueError("Missing or incompatible diagnostic score orientation")
    input_hash = marker.get("input_array_sha256", "")
    if len(input_hash) != 64 or any(c not in "0123456789abcdef" for c in input_hash):
        raise ValueError("Missing input-array fingerprint")
    frames = {}
    for kind, name in marker["files"].items():
        path = output / name
        if marker.get("sha256", {}).get(kind) != common.sha256_file(path):
            raise ValueError(f"Checksum mismatch: {name}")
        frames[kind] = pd.read_csv(path, float_precision="round_trip")

    metadata = marker.get("repetition_metadata", [])
    states = config["random_states"]
    if len(metadata) != len(states) or [m["random_state"] for m in metadata] != states:
        raise ValueError("Invalid repetition metadata")
    by_state = {m["random_state"]: m for m in metadata}
    for kind, frame in frames.items():
        if not {"dataset", "random_state"}.issubset(frame):
            raise ValueError(f"Missing identifiers in {kind}")
        if frame["dataset"].isna().any() or not frame["dataset"].eq(config["dataset"]).all():
            raise ValueError(f"Dataset mismatch in {kind}")
        if frame["random_state"].isna().any() or set(frame["random_state"]) != set(states):
            raise ValueError(f"Repetition mismatch in {kind}")
    score, roc, auroc, operating = (frames[k] for k in ("scores", "roc", "auroc", "operating"))
    if tuple(score.columns) != SCORE_COLUMNS or tuple(roc.columns) != ROC_COLUMNS:
        raise ValueError("Unexpected score or ROC schema")
    if score.duplicated(["random_state", "partition", "sample_index"]).any():
        raise ValueError("Duplicate observation score rows")
    if not np.isfinite(score[list(common.METHODS)].to_numpy()).all():
        raise ValueError("Nonfinite observation scores")
    if score[["sample_index", "label"]].isna().any().any() or (score["sample_index"] < 0).any():
        raise ValueError("Invalid observation index or class label")
    if not np.equal(score["sample_index"], np.floor(score["sample_index"])).all():
        raise ValueError("Observation indices must be integers")
    if not np.equal(score["label"], np.floor(score["label"])).all():
        raise ValueError("Class labels must be integers")
    expected_pairs = {(state, method) for state in states for method in common.METHODS}
    if len(auroc) != len(expected_pairs) or auroc.duplicated(["random_state", "method"]).any():
        raise ValueError("Missing or duplicate AUROC rows")
    if set(zip(auroc["random_state"], auroc["method"])) != expected_pairs:
        raise ValueError("Unexpected AUROC method/state grid")
    if set(zip(roc["random_state"], roc["method"])) != expected_pairs:
        raise ValueError("Missing ROC method/state curves")
    expected_op = {(s, m, float(a)) for s, m in expected_pairs for a in config["alpha_grid"]}
    if len(operating) != len(expected_op) or operating.duplicated(["random_state", "method", "alpha"]).any():
        raise ValueError("Missing or duplicate operating-point rows")
    if set(zip(operating["random_state"], operating["method"], operating["alpha"])) != expected_op:
        raise ValueError("Unexpected operating-point grid")
    spec = common.EXPERIMENTS[config["dataset"]]
    for state in states:
        rows = score[score["random_state"] == state]
        meta = by_state[state]
        if set(rows["partition"]) != set(PARTITIONS) or rows["sample_index"].duplicated().any():
            raise ValueError("Missing or overlapping held-out partitions")
        for partition, size_key in zip(PARTITIONS, ("n_cal", "n_test", "n_ood")):
            part = rows[rows["partition"] == partition]
            if int(meta[size_key]) <= 0 or len(part) != meta[size_key]:
                raise ValueError("Observation counts disagree with metadata")
            if common.class_counts(part["label"].to_numpy()) != meta["class_counts"][partition]:
                raise ValueError("Observation class counts disagree with metadata")
            intended = set(spec.labels_ood if partition == "ood" else spec.labels_id)
            observed = set(part["label"])
            if not observed.issubset(intended) or partition == "ood" and observed != intended:
                raise ValueError("Unexpected ID/OOD labels")
        for method in common.METHODS:
            curve = roc[(roc["random_state"] == state) & (roc["method"] == method)]
            rates = curve[["id_false_rejection", "ood_rejection"]].to_numpy()
            thresholds = curve["detection_threshold"].to_numpy()
            if (len(curve) < 2 or not np.isfinite(rates).all() or
                    (rates < 0).any() or (rates > 1).any() or
                    not np.array_equal(rates[0], [0., 0.]) or
                    not np.array_equal(rates[-1], [1., 1.]) or (np.diff(rates, axis=0) < 0).any() or
                    np.isnan(thresholds).any() or not np.isfinite(thresholds[1:]).all() or
                    not (np.diff(thresholds) < 0).all()):
                raise ValueError("Invalid ROC curve ordering or coordinates")
            metric = auroc[(auroc["random_state"] == state) & (auroc["method"] == method)].iloc[0]
            if not np.isclose(auc(rates[:, 0], rates[:, 1]), metric["ood_auroc"], rtol=0, atol=2e-12):
                raise ValueError("ROC integral and AUROC export disagree")
            for key in ("n_train", "n_cal", "n_test", "n_ood"):
                if metric[key] != meta[key] or meta[key] <= 0:
                    raise ValueError("AUROC split counts disagree with metadata")
            if metric["dimension"] != config["dimension"] or metric["temperature"] != config["temperature"]:
                raise ValueError("AUROC configuration disagrees with metadata")
            for _, op in operating[(operating["random_state"] == state) & (operating["method"] == method)].iterrows():
                threshold = common.lower_tail_threshold(rows.loc[rows["partition"] == "calibration", method].to_numpy(), op["alpha"])
                if threshold != op["threshold"]:
                    raise ValueError("Exported cutoff disagrees with ID-calibration scores")
                for partition, count_key, rate_key, size_key in (
                        ("test", "id_rejected", "id_false_rejection", "n_test"),
                        ("ood", "ood_rejected", "ood_rejection", "n_ood")):
                    rejected = int((rows.loc[rows["partition"] == partition, method] < threshold).sum())
                    if (op[count_key] != rejected or op[size_key] != meta[size_key] or
                            not np.isclose(op[rate_key], rejected / meta[size_key], rtol=0, atol=1e-12)):
                        raise ValueError("Exported rejection counts/rates disagree with observation scores")
    return True


def is_complete(marker_path, config):
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        return validate_export(marker, marker_path.parent, config)
    except (OSError, ValueError, KeyError, TypeError, IndexError, pd.errors.ParserError):
        return False


def publish_new(temporary, target):
    """Publish only a new diagnostic artifact while holding the group lock."""
    if target.exists():
        raise FileExistsError(f"Refusing to replace an existing diagnostic artifact: {target}")
    temporary.replace(target)


def main():
    args = parser().parse_args()
    if min(args.seed_group, args.repetitions, args.dimension, args.threads) < 1:
        raise ValueError("Seed group, repetitions, dimension and threads must be positive")
    if args.temperature <= 0 or not np.isfinite(args.temperature):
        raise ValueError("Energy temperature must be positive and finite")
    args.alpha_grid = sorted(set(round(float(a), 12) for a in (args.alpha_grid or (
        common.EXPERIMENTS[args.dataset].nominal_alpha * common.ALPHA_MULTIPLIERS).tolist())))
    if not args.alpha_grid or any(not np.isfinite(a) or not 0 < a < 1 for a in args.alpha_grid):
        raise ValueError("Alpha grid must contain finite values in (0,1)")
    output = output_folder(args)
    config = run_config(args.dataset, args)
    prefix = f"seed{args.seed_group}"
    marker_path = output / f"{prefix}_complete.json"
    if args.check_complete:
        raise SystemExit(0 if is_complete(marker_path, config) else 1)
    if is_complete(marker_path, config):
        print(f"Already complete: {marker_path}", flush=True)
        return
    if list(output.glob(f"{prefix}_*")):
        raise FileExistsError(f"Existing incomplete or incompatible diagnostic artifacts under {output}/{prefix}_*; choose a new results root or review them explicitly. Nothing was replaced.")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / f"{prefix}_running.lock"
    temporary_paths = []
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    try:
        existing = [p for p in output.glob(f"{prefix}_*") if p != lock_path]
        if existing:
            raise FileExistsError(f"Diagnostic artifacts appeared while acquiring the lock: {existing}")
        torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        print(f"Dataset={args.dataset}; seed group={args.seed_group}; reps={args.repetitions}; data={args.data_root}; diagnostics={output}; device={args.device}", flush=True)
        X, y = common.load_inputs(args.dataset, args.data_root)
        input_hash = common.array_digest(X.numpy() if isinstance(X, torch.Tensor) else X, y)
        groups = {key: [] for key in file_names(args.seed_group)}
        repetitions = []
        for state in config["random_states"]:
            print(f"Starting repetition {state}", flush=True)
            with threadpool_limits(limits=args.threads):
                frames, metadata = export_repetition(args.dataset, X, y, state, args)
            for kind, frame in frames.items():
                groups[kind].append(frame)
            repetitions.append(metadata)
            print(f"Finished repetition {state}: ID test={metadata['n_test']}, OOD={metadata['n_ood']}", flush=True)
        files = file_names(args.seed_group)
        for kind, name in files.items():
            target = output / name
            temporary = output / f"{name}.tmp.{os.getpid()}"
            temporary_paths.append(temporary)
            compression = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
            pd.concat(groups[kind], ignore_index=True).to_csv(temporary, index=False, compression=compression)
            publish_new(temporary, target)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                               text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        marker = {**config, "run_config": config, "files": files,
                  "sha256": {kind: common.sha256_file(output / name) for kind, name in files.items()},
                  "input_array_sha256": input_hash, "repetition_metadata": repetitions,
                  "git_revision": revision, "score_direction": "larger means ID",
                  "energy_input": "raw cosine in [-1,1]", "roc_positive_class": "ood",
                  "roc_score": "-id_score", "roc_comparison": ">= detection_threshold",
                  "threshold_rule": "id_score < threshold",
                  "calibration_rule": "k=floor(alpha*(n_cal+1)); kth ascending ID-calibration score; -inf when k=0"}
        validate_export(marker, output, config)
        temporary = output / f"{prefix}_complete.json.tmp.{os.getpid()}"
        temporary_paths.append(temporary)
        temporary.write_text(json.dumps(marker, indent=2, allow_nan=False), encoding="utf-8")
        publish_new(temporary, marker_path)
        print(f"Completed: {marker_path}", flush=True)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
