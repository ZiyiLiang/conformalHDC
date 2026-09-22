''' 
    Full conformal prediction, set-valued CHDC with marginal coverage 
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
