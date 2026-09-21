''' 
    Full conformal inference
    Marginal prediction sets for encoded hypervectors.
    Each candidate label triggers augmented-data prototype training,
    scoring of all n+1 observations 
'''
import numpy as np

from .methods import ConformalHDC


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

    def _validate_HVs(self, HVs, name):
        ''' Validate a matrix without changing its numeric precision. '''
        HVs = np.asarray(HVs)
        if HVs.ndim != 2 or HVs.shape[1] == 0:
            raise ValueError(f"{name} must have shape (n_samples, dimension > 0).")
        if HVs.dtype.kind not in "biufc" or not np.isfinite(HVs).all():
            raise ValueError(f"{name} must contain finite numeric values.")
        if self.sim_measure == "cosine" and np.iscomplexobj(HVs):
            raise ValueError("Use complex_cosine or euclidean for complex hypervectors.")
        return HVs

    def fit(self, dev_HVs, dev_labels, score_type="discount", **kwargs):
        ''' Store the complete development set and scoring configuration.

            There is no calibration split. Fitting each augmented classifier is
            deferred until its test hypervector and candidate label are available.
        '''
        HVs = self._validate_HVs(dev_HVs, "dev_HVs")
        labels = np.asarray(dev_labels, dtype=object)
        self.dev_HVs = HVs.copy()
        self.dev_labels = labels.copy()
        self.dev_targets = np.array([self.label_to_idx[label] for label in labels])
        self.n_development, self.hv_dimension = HVs.shape
        self.score_type = score_type
        self.score_kwargs = dict(kwargs)
        return self

    def _candidate_scores(self, aug_HVs, aug_labels, targets):
        ''' Refit one augmented classifier and score all n+1 observations with it. '''
        prototypes = self._validate_HVs(
            self.prototype_builder(aug_HVs, aug_labels, self.class_labels),
            "prototypes",
        )
        expected_shape = (len(self.class_labels), self.hv_dimension)
        if prototypes.shape != expected_shape:
            raise ValueError(f"prototype_builder must return shape {expected_shape}.")

        model = ConformalHDC(prototypes, self.class_labels,
                             sim_measure=self.sim_measure,
                             random_state=self.random_state, verbose=self.verbose)
        # Preserve methods.py's score formulas and seeded randomization. One call
        # scores the entire augmented set, including its final test observation.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scores = model._compute_nonconformity_scores(
                aug_HVs, targets, score_type=self.score_type, **self.score_kwargs,
            )
        if not np.isfinite(scores).all():
            raise ValueError("Nonconformity scores must be finite; check prototypes and score settings.")
        return scores

    def set_valued_CP(self, test_HVs, alpha):
        ''' Return marginal prediction sets using Algorithm 1's exact order statistic.

            Args:
                test_HVs: Encoded samples (n_test, dim), or one (dim,) sample.
                alpha: Significance level, 0 <= alpha < 1.
            Returns:
                One list of real labels per test sample, in class_labels order.
                Empty sets are retained. Each test sample is augmented separately.
        '''
        if not hasattr(self, "dev_HVs"):
            raise RuntimeError("Fit the development set first! (Call function fit)")

        test_HVs = self._validate_HVs(np.atleast_2d(test_HVs), "test_HVs")

        n_aug = self.n_development + 1
        rank = int(np.ceil((1 - alpha) * n_aug)) - 1
        aug_HVs = np.empty((n_aug, self.hv_dimension),
                           dtype=np.result_type(self.dev_HVs.dtype, test_HVs.dtype))
        aug_HVs[:-1] = self.dev_HVs
        aug_labels = np.empty(n_aug, dtype=object)
        aug_labels[:-1] = self.dev_labels
        targets = np.empty(n_aug, dtype=int)
        targets[:-1] = self.dev_targets

        psets = [[] for _ in range(len(test_HVs))]
        for i, HV in enumerate(test_HVs):
            aug_HVs[-1] = HV
            for c_idx, label in enumerate(self.class_labels):
                aug_labels[-1] = label
                targets[-1] = c_idx
                scores = self._candidate_scores(aug_HVs, aug_labels, targets)
                quantile = np.partition(scores, rank)[rank]
                if scores[-1] <= quantile:
                    psets[i].append(label)
        return psets
