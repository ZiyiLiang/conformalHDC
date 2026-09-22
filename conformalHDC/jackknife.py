'''
    Jackknife+ set-valued CHDC with marginal coverage.
    Fit one classifier per held-out development sample, then compare its test
    scores with the score it assigns to that held-out sample (Algorithm 2).
'''
import numpy as np

from .methods import ConformalHDC


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
        n_test = len(test_HVs)
        counts = np.zeros((n_test, len(self.class_labels)), dtype=int)
        for model, heldout_score in zip(self.loo_models, self.loo_scores):
            sims = model._sim_matrix(test_HVs)
            for c_idx in range(len(self.class_labels)):
                scores = model._scores_from_sims(
                    sims, np.full(n_test, c_idx),
                    score_type=self.score_type, **self.score_kwargs,
                ) 
                counts[:, c_idx] += scores > heldout_score

        threshold = np.ceil((1 - alpha) * (self.n_development + 1))
        return [[label for label, keep in zip(self.class_labels, row) if keep]
                for row in counts < threshold]
