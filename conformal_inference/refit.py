"""Reference jackknife+ and full conformal prediction that refit every model.

Works with any prototype_builder but is slow: jackknife+ fits one classifier per
held-out development sample and compares its test scores with the score it assigns
to that sample (Algorithm 2); full conformal fits one classifier per test sample and
candidate label on the augmented data and scores all n + 1 observations.
conformalHDC.fast gives the same sets for class-sum prototypes without refitting.
"""
import numpy as np

from .methods import ConformalHDC, sets_from_mask


class JackknifePlusHDC():
    def __init__(self, class_labels, prototype_builder, sim_measure="cosine",
                 random_state=0, verbose=True):
        if not callable(prototype_builder):
            raise TypeError("prototype_builder must be callable.") 
        
        self.class_labels = list(class_labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.class_labels)}
        if not self.class_labels or len(self.label_to_idx) != len(self.class_labels):
            raise ValueError("class_labels must be nonempty and unique.")
        self.prototype_builder = prototype_builder
        self.sim_measure = sim_measure
        self.random_state = random_state
        self.verbose = verbose
        self.loo_models = None
        self._is_fitted = False

    def _build_model(self, HVs, labels):
        ''' Build prototypes in class_labels order for one leave-one-out dataset.

            The builder must also return a prototype for classes absent from the
            subset, using its own missing-class convention.
        '''
        prototypes = np.asarray(self.prototype_builder(HVs, labels, self.class_labels))

        return ConformalHDC(prototypes.copy(), self.class_labels,
                            sim_measure=self.sim_measure,
                            random_state=self.random_state, verbose=self.verbose)

    def fit(self, dev_HVs, dev_labels, score_type="discount", **kwargs):
        ''' Fit each leave-one-out classifier on already encoded hypervectors. '''
        self._is_fitted = False
        self.loo_models = None
 
        self.dev_HVs = dev_HVs.copy()
        self.dev_labels = dev_labels.copy()
        self.dev_targets = np.array([self.label_to_idx[label] for label in dev_labels])
        self.n_development, self.hv_dimension = dev_HVs.shape
        models = []
        keep = np.ones(self.n_development, dtype=bool)
        for i in range(self.n_development):
            keep[i] = False
            models.append(self._build_model(self.dev_HVs[keep], self.dev_labels[keep]))
            keep[i] = True
        self.loo_models = models
        return self.set_score_type(score_type, **kwargs)

    def set_score_type(self, score_type, **kwargs):
        ''' Rescore held-out samples without rebuilding the leave-one-out models. '''
        if self.loo_models is None:
            raise RuntimeError("Fit the development set first! (Call function fit)")
        scores = np.array([
            model._compute_nonconformity_scores(
                self.dev_HVs[i:i + 1], self.dev_targets[i:i + 1],
                score_type=score_type, **kwargs,
            )[0]
            for i, model in enumerate(self.loo_models)
        ]) 
        self.loo_scores = scores
        self.score_type = score_type
        self.score_kwargs = dict(kwargs)
        self._is_fitted = True
        return self

    def set_valued_CP(self, test_HVs, alpha):
        ''' Return marginal prediction sets using Algorithm 2's strict comparisons.

            test_HVs accepts (n_test, dim) or one (dim,) hypervector.
            Labels retain class_labels order; empty prediction sets are retained.
            Randomized scores follow methods.py's seeded draws for each call.
        ''' 
        if self.loo_models is None:
            raise RuntimeError("Fit the development set first! (Call function fit)")
        counts = np.zeros((len(test_HVs), len(self.class_labels)), dtype=int)
        for model, heldout_score in zip(self.loo_models, self.loo_scores):
            counts += model._class_scores(
                model._sim_matrix(test_HVs), self.score_type, **self.score_kwargs,
            ) > heldout_score

        threshold = np.ceil((1 - alpha) * (self.n_development + 1))
        return sets_from_mask(counts < threshold, self.class_labels)


def jackknife_all_scores(template, H, y, X, alpha, score_types):
    """Consume one leave-one-out model at a time across all score types."""
    labels = template.class_labels
    targets = np.array([template.label_to_idx[label] for label in y])
    counts = {
        s: np.zeros((len(X), len(labels)), dtype=np.int64)
        for s in score_types
    }
    keep = np.ones(len(H), dtype=bool)

    for i in range(len(H)):
        keep[i] = False
        model = template._build_model(H[keep], y[keep])
        keep[i] = True
        held_sims = model._sim_matrix(H[i:i + 1])
        test_sims = model._sim_matrix(X)

        for s in score_types:
            held_score = model._scores_from_sims(
                held_sims, targets[i:i + 1], score_type=s,
            )[0]
            counts[s] += model._class_scores(test_sims, s) > held_score

    threshold = np.ceil((1 - alpha) * (len(H) + 1))
    return {
        s: sets_from_mask(counts[s] < threshold, labels)
        for s in score_types
    }


class FullConformalHDC():

    def __init__(self, class_labels, prototype_builder, sim_measure="cosine",
                 random_state=0, verbose=True):
        if not callable(prototype_builder):
            raise TypeError("prototype_builder must be callable.")

        self.class_labels = list(class_labels)
        self.label_to_idx = {label: i for i, label in enumerate(self.class_labels)}
        self.prototype_builder = prototype_builder
        self.sim_measure = sim_measure
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, dev_HVs, dev_labels, score_type="discount", **kwargs):
        ''' Store the complete development set and scoring configuration.

            There is no calibration split. Fitting each augmented classifier is
            deferred until its test hypervector and candidate label are available.
        ''' 
        labels = np.asarray(dev_labels, dtype=object)
        self.dev_HVs = dev_HVs.copy()
        self.dev_labels = labels.copy()
        self.dev_targets = np.array([self.label_to_idx[label] for label in labels])
        self.n_development, self.hv_dimension = dev_HVs.shape
        self.score_type = score_type
        self.score_kwargs = dict(kwargs)
        return self

    def _candidate_scores(self, aug_HVs, aug_labels, targets):
        ''' Refit one augmented classifier and score all n+1 observations with it. '''

        prototypes =  self.prototype_builder(aug_HVs, aug_labels, self.class_labels) 
        model = ConformalHDC(prototypes, self.class_labels,
                             sim_measure=self.sim_measure,
                             random_state=self.random_state, verbose=self.verbose)
        scores = model._compute_nonconformity_scores(
                aug_HVs, targets, score_type=self.score_type, **self.score_kwargs,
            )
        return scores

    def set_valued_CP(self, test_HVs, alpha):
        ''' Return marginal prediction sets using cv plus.

            Args:
                test_HVs: Encoded samples (n_test, dim), or one (dim,) sample.
                alpha: Significance level, 0 <= alpha < 1.
            Returns:
                One list of real labels per test sample, in class_labels order.
                Empty sets are retained. Each test sample is augmented separately.
        ''' 

        n_aug = self.n_development + 1
        Qrank = int(np.ceil((1 - alpha) * n_aug)) - 1 # (1-alpha)(n+1)-th
        aug_HVs = np.empty((n_aug, self.hv_dimension),
                           dtype=np.result_type(self.dev_HVs.dtype, test_HVs.dtype))
        aug_HVs[:-1] = self.dev_HVs # train + cal
        aug_labels = np.empty(n_aug, dtype=object)
        aug_labels[:-1] = self.dev_labels
        targets = np.empty(n_aug, dtype=int)
        targets[:-1] = self.dev_targets

        psets = [[] for _ in range(len(test_HVs))]
        for i, HV in enumerate(test_HVs):
            aug_HVs[-1] = HV
            for c_idx, label in enumerate(self.class_labels):
                # assign the test feature to class c
                aug_labels[-1] = label
                targets[-1] = c_idx
                scores = self._candidate_scores(aug_HVs, aug_labels, targets)
                Q = np.partition(scores, Qrank)[Qrank] # (1-alpha)(n+1)-th smallest value
                if scores[-1] <= Q:
                    psets[i].append(label)
        return psets

def full_conformal_all_scores(template, H, y, X, alpha, score_types):
    """Reuse each augmented model and its similarities across score types."""
    labels = template.class_labels
    aug = np.empty(
        (len(H) + 1, H.shape[1]),
        dtype=np.result_type(H.dtype, X.dtype),
    )
    aug[:-1] = H
    aug_y = np.empty(len(H) + 1, dtype=object)
    aug_y[:-1] = y
    targets = np.empty(len(H) + 1, dtype=int)
    targets[:-1] = [template.label_to_idx[label] for label in y]
    rank = int(np.ceil((1 - alpha) * len(aug))) - 1
    result = {s: [[] for _ in X] for s in score_types}

    for i, x in enumerate(X):
        aug[-1] = x
        for c, label in enumerate(labels):
            aug_y[-1], targets[-1] = label, c
            prototypes = template.prototype_builder(aug, aug_y, labels)
            model = ConformalHDC(
                prototypes, labels, sim_measure=template.sim_measure,
                random_state=template.random_state, verbose=template.verbose,
            )
            sims = model._sim_matrix(aug)
            for s in score_types:
                scores = model._scores_from_sims(sims, targets, score_type=s)
                if scores[-1] <= np.partition(scores, rank)[rank]:
                    result[s][i].append(label)

    return result
