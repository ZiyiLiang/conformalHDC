"""Fast jackknife+ and full conformal prediction for class-sum prototypes.

Leaving out a development sample, or adding a test sample to class c, changes only
class c's sum, hence only prototype c and similarity column c. A prototype rule
computes that column without refitting; FastConformal turns columns into sets.

Rules ("bipolar", "ternary", "normalized"):
    bipolar     sign of the class sum, a zero sum mapping to +1 (MNIST)
    ternary     np.sign of the class sum, a zero sum staying 0 (HAR, ISOLET)
    normalized  class sum scaled to unit norm (Languages)
"""
import multiprocessing
from collections import defaultdict

import numpy as np
from threadpoolctl import threadpool_limits

from .methods import ConformalHDC, sets_from_mask


################======== Prototypes ========################

def label_targets(y, labels):
    """Canonical class index of every label in y; labels must be unique and cover y."""
    label_to_idx = {label: c for c, label in enumerate(labels)}
    if not len(labels) or len(label_to_idx) != len(labels):
        raise ValueError("Labels must be nonempty and unique.")
    if any(label not in label_to_idx for label in y):
        raise ValueError("Labels in y must belong to the supplied classes.")
    return np.array([label_to_idx[label] for label in y], dtype=int)


def class_sums(H, targets, n_classes):
    """(n_classes, dim) sums of the rows of H per class; exact for bipolar float32 rows."""
    if len(H) >= 2**24:
        raise ValueError("Too many rows for exact float32 class sums.")
    one_hot = (np.asarray(targets)[:, None] == np.arange(n_classes)).astype(H.dtype)
    return one_hot.T @ H


def sign_prototypes(sums, zero=1):
    """Sign of the class sums as float32, mapping a zero sum to `zero`."""
    return np.where(sums > 0, 1, np.where(sums < 0, -1, zero)).astype(np.float32)


def unit_rows(sums):
    """Rows scaled to unit norm; all-zero rows stay zero."""
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    return np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 1e-8)


def class_prototypes(H, y, labels, rule):
    """Prototypes of H in labels order under `rule` (see the module docstring)."""
    sums = class_sums(H, label_targets(y, labels), len(labels))
    return unit_rows(sums) if rule == "normalized" else sign_prototypes(sums, ZERO[rule])


################======== Prototype rules ========################

class SignRule:
    """Sign prototypes of bipolar float32 data.

    A sum moving by one changes the sign only where it is within one of zero, so a
    model differs from the base prototype at a few coordinates. Dots are integers,
    so similarities are exact; they often tie, so every model is scored exactly.
    """
    batched = False

    def __init__(self, H, X, targets, sums, zero):
        for data in (H, X):  # checked in row blocks to bound the temporaries
            if data.dtype != np.float32 or not all(np.all(np.abs(b) == 1) for b in chunks(data, 4096)):
                raise ValueError("The sign rule requires bipolar (+1/-1) float32 hypervectors.")
        if H.shape[1] >= 2**22:
            raise ValueError("Dimension too large for exact float32 dot corrections.")
        self.data, self.targets, self.zero = {"dev": H, "test": X}, targets, zero
        self.sums = sums.astype(np.int32)
        self.prototypes = sign_prototypes(self.sums, zero)
        # Coordinates whose sign changes when the sum moves by one
        self.critical = [
            np.flatnonzero((sign_prototypes(s + 1, zero) != p) | (sign_prototypes(s - 1, zero) != p))
            for s, p in zip(self.sums, self.prototypes)
        ]
        # All rows share one norm; denominators follow _sim_matrix's float32 arithmetic.
        self.hv_norm = np.linalg.norm(H[:1], axis=1)[0]
        self.proto_sq = (self.prototypes ** 2).sum(axis=1)
        denominators = self.hv_norm * np.linalg.norm(self.prototypes, axis=1) + 1e-8
        self.dots = {w: x @ self.prototypes.T for w, x in self.data.items()}
        self.sims = {w: self._similarities(d, denominators) for w, d in self.dots.items()}

    @staticmethod
    def _similarities(dots, denominators):
        return np.asarray((1 + dots / denominators) / 2, dtype=float)

    def _change(self, c, sample, direction):
        """(c, columns, deltas) of prototype c when sample is added (+1) or removed (-1)."""
        columns = self.critical[c]
        old = self.prototypes[c, columns]
        new = sign_prototypes(self.sums[c, columns] + direction * sample[columns], self.zero)
        changed = new != old
        return c, tuple(columns[changed]), tuple((new - old)[changed])

    def leave_one_out(self, i):
        return self._change(self.targets[i], self.data["dev"][i], -1)

    def augment(self, j, c):
        return self._change(c, self.data["test"][j], 1)

    def column(self, change, which, rows):
        """Similarities of the `which` ("dev"/"test") rows to prototype c after `change`."""
        c, columns, delta = change
        if not columns:
            return self.sims[which][rows, c]
        columns, delta = np.array(columns), np.array(delta, dtype=np.float32)
        old = self.prototypes[c, columns]
        sq = self.proto_sq[c] + np.sum((old + delta) ** 2 - old ** 2)
        denominator = self.hv_norm * np.sqrt(sq) + np.float32(1e-8)
        dots = self.dots[which][rows, c] + self.data[which][:, columns][rows] @ delta
        return self._similarities(dots, denominator)


class NormalizedRule:
    """Unit-norm class-sum prototypes.

    Moving class sum c by a sample moves every dot with it by that sample's Gram
    entries, so all models follow from the base dots and one dev-test Gram block.
    Similarities rarely tie, so models are scored in batches.
    """
    batched = True

    def __init__(self, H, X, targets, sums):
        self.data, self.targets = {"dev": H, "test": X}, targets
        self.prototypes = unit_rows(sums)
        self.sum_sq = (sums.astype(float) ** 2).sum(axis=1)                      # ||s_c||^2
        self.sq = {w: np.einsum("nd,nd->n", x, x).astype(float) for w, x in self.data.items()}
        self.dots = {w: (x @ sums.T).astype(float) for w, x in self.data.items()}
        self.gram = H @ X.T                                                       # h_i . x_j
        self.sims = {w: self._cosine(self.dots[w], self.sq[w][:, None], self.sum_sq) for w in self.data}

    @staticmethod
    def _cosine(dots, sample_sq, sum_sq):
        """(1 + cos) / 2 between samples and class sums; 1/2 for an empty class."""
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = (1 + dots / np.sqrt(sample_sq * sum_sq)) / 2
        return np.where(sum_sq > 0, sims, 0.5)

    def leave_one_out(self, i):
        return self.targets[i], -1, i

    def augment(self, j, c):
        return c, 1, j

    def column(self, change, which, rows):
        """Similarities of the `which` ("dev"/"test") rows to prototype c after `change`."""
        c, direction, i = change
        if direction > 0:  # test sample i joins class c
            cross = self.gram[rows, i] if which == "dev" else self.data["test"][rows] @ self.data["test"][i]
            sum_sq = self.sum_sq[c] + 2 * self.dots["test"][i, c] + self.sq["test"][i]
        else:              # development sample i leaves class c
            cross = self.gram[i, rows] if which == "test" else self.data["dev"][rows] @ self.data["dev"][i]
            sum_sq = self.sum_sq[c] - 2 * self.dots["dev"][i, c] + self.sq["dev"][i]
        return self._cosine(self.dots[which][rows, c] + direction * cross, self.sq[which][rows], sum_sq)


ZERO = {"bipolar": 1, "ternary": 0}
RULES = {
    "bipolar": lambda H, X, t, s: SignRule(H, X, t, s, zero=1),
    "ternary": lambda H, X, t, s: SignRule(H, X, t, s, zero=0),
    "normalized": NormalizedRule,
}


################======== Conformal engine ========################

def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


# Engine shared with forked worker processes (copy-on-write, so nothing large is copied).
_ENGINE = None


def _run_unit(task):
    method, args = task
    return getattr(_ENGINE, method)(*args)


class FastConformal:
    """Jackknife+ and full-conformal prediction sets without refitting any model.

    H, y: development set (train + calibration); X: test hypervectors.
    n_jobs: worker processes for the independent models; results do not depend on it.
    Input arrays must not be mutated while this object is in use.
    """

    def __init__(self, H, y, X, labels, rule, random_state=0, chunk_size=128, n_jobs=1):
        if H.ndim != 2 or X.ndim != 2 or not len(H) or X.shape[1] != H.shape[1]:
            raise ValueError("Development data must be nonempty with matching dimensions.")
        if len(y) != len(H):
            raise ValueError("Development labels must match the development data.")
        self.labels = list(labels)
        self.targets = label_targets(y, self.labels)
        self.rule = RULES[rule](H, X, self.targets, class_sums(H, self.targets, len(self.labels)))
        self.prototypes = self.rule.prototypes
        self.model = ConformalHDC(self.prototypes, self.labels, random_state=random_state)
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs

    def _map(self, method, arg_list):
        """[self.method(*args) for args in arg_list], on n_jobs forked workers when n_jobs > 1."""
        if self.n_jobs <= 1 or len(arg_list) < 2:
            return [getattr(self, method)(*args) for args in arg_list]
        global _ENGINE
        _ENGINE = self
        try:
            # One BLAS thread per worker, so n_jobs workers use n_jobs cores.
            with multiprocessing.get_context("fork").Pool(
                self.n_jobs, initializer=threadpool_limits, initargs=(1,),
            ) as pool:
                return pool.map(_run_unit, [(method, args) for args in arg_list])
        finally:
            _ENGINE = None

    def _sims(self, which, rows, change):
        """Similarity rows of `which` ("dev"/"test") rows under the model `change`."""
        sims = self.rule.sims[which][rows].copy()
        sims[:, change[0]] = self.rule.column(change, which, rows)
        return sims

    @staticmethod
    def _groups(changes):
        """[(change, indices)] of identical models, in first-seen order."""
        groups = defaultdict(list)
        for i, change in enumerate(changes):
            groups[change].append(i)
        return [(change, np.array(rows)) for change, rows in groups.items()]

    def _batches(self, groups):
        """Groups in same-class batches, or one group per batch for an unbatched rule."""
        if not self.rule.batched:
            return [[group] for group in groups]
        by_class = defaultdict(list)
        for group in groups:
            by_class[group[0][0]].append(group)
        return [batch for same in by_class.values() for batch in chunks(same, self.chunk_size)]

    def jackknife_sets(self, alpha, score_types):
        """Jackknife+ sets; model i leaves out development sample i."""
        n, (n_test, n_classes) = len(self.targets), self.rule.sims["test"].shape
        counts = {s: np.zeros((n_test, n_classes), dtype=np.int64) for s in score_types}
        # One draw per development observation, retained across all models.
        held_draw = self.model._uniform_draws(n)
        batches = self._batches(self._groups(self.rule.leave_one_out(i) for i in range(n)))
        n_parts = 4 * self.n_jobs if self.n_jobs > 1 else 1  # interleaved shares balance the classes
        for part in self._map("_jackknife_counts", [
            (batches[i::n_parts], score_types, held_draw) for i in range(n_parts)
        ]):
            for s in score_types:
                counts[s] += part[s]
        threshold = np.ceil((1 - alpha) * (n + 1))
        return {s: sets_from_mask(counts[s] < threshold, self.labels) for s in score_types}

    def _jackknife_counts(self, batches, score_types, held_draw):
        """Per score type, how many held-out scores each candidate score exceeds."""
        n_test, n_classes = self.rule.sims["test"].shape
        counts = {s: np.zeros((n_test, n_classes), dtype=np.int64) for s in score_types}
        for batch in batches:
            rows = np.concatenate([r for _, r in batch])
            owner = np.repeat(np.arange(len(batch)), [len(r) for _, r in batch])
            held = np.vstack([self._sims("dev", r, change) for change, r in batch])
            for s in score_types:
                held_scores = self.model._scores_from_sims(
                    held, self.targets[rows], score_type=s, U=held_draw[rows],
                )
                for k, candidate in self._candidate_scores(batch, s):
                    if len(batch) == 1:  # one model: count by binary search
                        counts[s][:, k] += np.searchsorted(np.sort(held_scores), candidate[:, 0], side="left")
                    else:
                        counts[s][:, k] += (candidate[:, owner] > held_scores).sum(axis=1)
        return counts

    def _candidate_scores(self, batch, score_type):
        """(k, (n_test, len(batch)) scores of every test sample for candidate class k)."""
        n_test, n_classes = self.rule.sims["test"].shape
        if not self.rule.batched:
            (change, _), = batch
            scores = self.model._class_scores(self._sims("test", slice(None), change), score_type,
                U=self.model._uniform_draws(n_test, offset=len(self.targets)),
            )
            return [(k, scores[:, [k]]) for k in range(n_classes)]
        c = batch[0][0][0]
        new = np.column_stack([self.rule.column(change, "test", slice(None)) for change, _ in batch])
        draws = self.model._uniform_draws(n_test, offset=len(self.targets))
        return [(k, self.model._column_update_scores(
            self.rule.sims["test"], np.full(n_test, k), c, new, score_type, U=draws,
        )) for k in range(n_classes)]

    def full_conformal_sets(self, alpha, score_types):
        """Full-conformal sets; model (j, c) adds test sample j to class c.

        The test sample is the last of n + 1 scored rows. It is kept when its score is
        at most the rank-th smallest of all n + 1 scores, which holds exactly when it
        is at most the rank-th smallest of the n development scores (always, when
        rank == n).
        """
        if not 0 <= alpha < 1:
            raise ValueError("alpha must lie in [0, 1).")
        n, (n_test, n_classes) = len(self.targets), self.rule.sims["test"].shape
        rank = int(np.ceil((1 - alpha) * (n + 1))) - 1
        keep = {s: np.full((n_test, n_classes), rank >= n) for s in score_types}
        if rank >= n:
            return {s: sets_from_mask(keep[s], self.labels) for s in score_types}
        units = [
            (c, batch, score_types, rank)
            for c in range(n_classes)
            for batch in self._batches(self._groups(self.rule.augment(j, c) for j in range(n_test)))
        ]
        for (c, *_), results in zip(units, self._map("_full_conformal_keep", units)):
            for tests, kept in results:
                for s in score_types:
                    keep[s][tests, c] = kept[s]
        return {s: sets_from_mask(keep[s], self.labels) for s in score_types}

    def _full_conformal_keep(self, c, batch, score_types, rank):
        """[(tests, {score type: kept})] for the models of one same-class batch."""
        n = len(self.targets)
        draws = self.model._uniform_draws(n + 1)  # seeded, so identical in every unit
        thresholds = self._dev_thresholds(c, [change for change, _ in batch], score_types, draws[:n], rank)
        results = []
        for b, (change, tests) in enumerate(batch):
            last = self._sims("test", tests, change)
            results.append((tests, {s: self.model._scores_from_sims(
                last, np.full(len(tests), c), score_type=s, U=np.full(len(tests), draws[n]),
            ) <= thresholds[s][b] for s in score_types}))
        return results

    def _dev_thresholds(self, c, changes, score_types, draws, rank):
        """{score type: rank-th smallest development score under each model}."""
        if self.rule.batched:
            new = np.column_stack([self.rule.column(change, "dev", slice(None)) for change in changes])
            return {s: np.partition(np.ascontiguousarray(self.model._column_update_scores(
                self.rule.sims["dev"], self.targets, c, new, s, U=draws,
            ).T), rank, axis=1)[:, rank] for s in score_types}
        # Rescore every row: sign similarities tie often, and ties must be broken
        # exactly as _scores_from_sims does.
        thresholds = {s: [] for s in score_types}
        for change in changes:
            dev = self._sims("dev", slice(None), change)
            for s in score_types:
                scores = self.model._scores_from_sims(dev, self.targets, score_type=s, U=draws)
                thresholds[s].append(np.partition(scores, rank)[rank])
        return thresholds
