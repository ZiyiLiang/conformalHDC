'''
    Jackknife+ w/ Conformal Inference
    Combines training and calibration hypervectors into one development set,
    builds every leave-one-out classifier, and compares the test scores against
    the score each classifier assigns to its own held-out sample.
'''
import numpy as np

from .methods import ConformalHDC


class JackknifePlusHDC:
    """Apply exact leave-one-out Jackknife+ to a prototype-based HDC classifier."""

    # Rough cap on the (n_development, chunk, n_class) similarity block, counted in floats.
    _MAX_SIM_BLOCK = 8_000_000

    def __init__(self, class_labels, prototype_builder, sim_measure="cosine",
                 random_state=0):
        if not callable(prototype_builder):
            raise TypeError("prototype_builder must be callable.")
        if sim_measure not in ("cosine", "euclidean", "complex_cosine"):
            raise ValueError(f"Unknown similarity measure: {sim_measure}")

        self.class_labels = list(class_labels)
        self.prototype_builder = prototype_builder
        self.sim_measure = sim_measure
        self.random_state = random_state
        self._is_fitted = False

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Fit the development set first! (Call function fit)")

    def _build_prototypes(self, HVs, labels):
        """Run prototype_builder on one subset and check the shape it returns."""
        prototypes = np.asarray(self.prototype_builder(HVs, labels, self.class_labels))

        expected_shape = (len(self.class_labels), self.hv_dimension)
        if prototypes.shape != expected_shape:
            raise ValueError(
                f"prototype_builder must return shape {expected_shape}. Got {prototypes.shape}"
            )

        return prototypes

    def _build_model(self, HVs, labels):
        """Wrap one prototype_builder call in an HDC classifier."""
        return ConformalHDC(
            class_HVs=self._build_prototypes(HVs, labels),
            class_labels=self.class_labels,
            sim_measure=self.sim_measure,
            random_state=self.random_state,
            verbose=False,
        )

    def fit(self, dev_HVs, dev_labels,
            score_type="discount", **score_kwargs):
        """Build every leave-one-out classifier and compute its calibration score."""

        self._is_fitted = False
        self.dev_HVs, self.dev_labels =  dev_HVs, dev_labels 
        self.n_development, self.hv_dimension = self.dev_HVs.shape

        # Model trained on the whole development set. Serves point predictions and
        # settles ties between labels that the leave-one-out models rank equally.
        self.full_model = self._build_model(self.dev_HVs, self.dev_labels)

        n_class = len(self.class_labels)
        self.loo_prototypes = np.empty(
            (self.n_development, n_class, self.hv_dimension),
            dtype=self.full_model.class_HVs.dtype,
        )
        keep = np.ones(self.n_development, dtype=bool)
        for left_out_idx in range(self.n_development):
            keep[left_out_idx] = False
            self.loo_prototypes[left_out_idx] = self._build_prototypes(
                self.dev_HVs[keep], self.dev_labels[keep]
            )
            keep[left_out_idx] = True

        self._prepare_sim_cache()
        self._is_fitted = True
        return self.set_score_type(score_type, **score_kwargs)

    def set_score_type(self, score_type, **score_kwargs):
        """Recompute the calibration scores under a different nonconformity score.

        Prototypes do not depend on the score, so sweeping several score types over one
        development set costs a single fit plus one cheap rescore per score type.
        """
        self._check_fitted()
        self.score_type = score_type
        self.score_kwargs = dict(score_kwargs)

        targets = [self.full_model.label_to_idx[label] for label in self.dev_labels]
        # fit scored one sample per model, so every draw was the first of a fresh stream
        U = np.full(self.n_development,
                    np.random.RandomState(self.random_state).uniform(0, 1))
        self.loo_scores = self.full_model._scores_from_sims(
            self._holdout_sims(), targets,
            score_type=self.score_type, U=U, **self.score_kwargs,
        )
        return self

    def _holdout_sims(self):
        ''' Similarities between each held-out development sample and the prototypes of the
            model that left it out, i.e. the diagonal over leave-one-out models.

            Returns:
                shape (n_development, n_class)
        '''
        cross = np.einsum("nd,ncd->nc", self.dev_HVs, np.conj(self.loo_prototypes))

        if self.sim_measure == "cosine":
            dev_norms = np.linalg.norm(self.dev_HVs, axis=1)
            proto_norms = np.linalg.norm(self.loo_prototypes, axis=2)
            return (1 + cross / (dev_norms[:, None] * proto_norms + 1e-8)) / 2

        if self.sim_measure == "complex_cosine":
            return (1 + np.real(cross) / self.hv_dimension) / 2

        dev_sq_norms = np.sum(np.abs(self.dev_HVs) ** 2, axis=1)
        proto_sq_norms = np.sum(np.abs(self.loo_prototypes) ** 2, axis=2)
        distances = np.sqrt(np.clip(
            dev_sq_norms[:, None] + proto_sq_norms - 2 * np.real(cross), 0, None,
        ))
        return np.where(distances == 0, 1e8, 1.0 / np.where(distances == 0, 1, distances))

    def _prepare_sim_cache(self):
        """Flatten the leave-one-out prototypes once so similarities are a single matmul."""
        self._flat_prototypes = self.loo_prototypes.reshape(-1, self.hv_dimension)

        if self.sim_measure == "cosine":
            self._prototype_norms = np.linalg.norm(self._flat_prototypes, axis=1)
        elif self.sim_measure == "euclidean":
            self._prototype_sq_norms = np.sum(np.abs(self._flat_prototypes) ** 2, axis=1)

    def _loo_sims(self, test_chunk):
        ''' Similarities between a chunk of test HVs and every leave-one-out prototype.

            Returns:
                shape (n_development * chunk, n_class). Row m*chunk + i holds the
                similarities of test HV i under the model that left out development
                sample m, which is the layout _scores_from_sims expects.
        '''
        flat = self._flat_prototypes

        if self.sim_measure == "cosine":
            dots = test_chunk @ flat.T
            test_norms = np.linalg.norm(test_chunk, axis=1)
            sims = dots / (np.outer(test_norms, self._prototype_norms) + 1e-8)
            sims = (1 + sims) / 2  # ranges from [0,1]

        elif self.sim_measure == "complex_cosine":
            sims = np.real(test_chunk @ np.conj(flat).T) / self.hv_dimension
            sims = (1 + sims) / 2  # ranges from [0,1]

        else:  # euclidean, via ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            cross = np.real(test_chunk @ np.conj(flat).T)
            test_sq_norms = np.sum(np.abs(test_chunk) ** 2, axis=1)
            distances = np.sqrt(np.clip(
                test_sq_norms[:, None] + self._prototype_sq_norms[None, :] - 2 * cross,
                0, None,
            ))
            sims = np.where(distances == 0, 1e8, 1.0 / np.where(distances == 0, 1, distances))

        n_class = len(self.class_labels)
        sims = sims.reshape(len(test_chunk), self.n_development, n_class)
        return sims.transpose(1, 0, 2).reshape(-1, n_class)

    def _auto_chunk_size(self):
        """Pick a test chunk that keeps the similarity block to a bounded size."""
        return max(1, self._MAX_SIM_BLOCK // (self.n_development * len(self.class_labels)))

    def _comparison_counts(self, test_HVs, chunk_size=None):
        ''' Counts, per test HV and candidate label, how many leave-one-out models score
            that label worse than they score their own held-out development sample.

            A low count means the label is conformal: few models argue against it.
        '''
        self._check_fitted()
        test_HVs = np.atleast_2d(np.asarray(test_HVs))

        n_test = len(test_HVs)
        n_class = len(self.class_labels)
        counts = np.zeros((n_test, n_class), dtype=np.int64)
        chunk_size = chunk_size or self._auto_chunk_size()

        # One uniform per test sample, reused by every leave-one-out model, so the
        # randomized scores stay independent of the chunking.
        randomizer = np.random.RandomState(self.random_state).uniform(0, 1, size=n_test)

        for start in range(0, n_test, chunk_size):
            chunk = test_HVs[start:start + chunk_size]
            sims = self._loo_sims(chunk)
            chunk_U = np.tile(randomizer[start:start + len(chunk)], self.n_development)

            for candidate_idx in range(n_class):
                candidate_scores = self.full_model._scores_from_sims(
                    sims,
                    np.full(len(sims), candidate_idx),
                    score_type=self.score_type,
                    U=chunk_U,
                    **self.score_kwargs,
                ).reshape(self.n_development, len(chunk))

                counts[start:start + len(chunk), candidate_idx] = np.sum(
                    candidate_scores > self.loo_scores[:, None], axis=0
                )

        return counts

    def _inclusion_mask(self, counts, alpha):
        """Jackknife+ inclusion rule: keep the labels too few models rejected."""
 
        inclusion_threshold = np.ceil((1 - alpha) * (self.n_development + 1))
        return counts < inclusion_threshold

    def set_valued_CP(self, test_HVs, alpha, allow_empty=True):
        """Create marginal Jackknife+ prediction sets for encoded test samples."""

        mask = self._inclusion_mask(self._comparison_counts(test_HVs), alpha)

        pred_sets = [
            [label for label, keep in zip(self.class_labels, row) if keep]
            for row in mask
        ] 
        if not allow_empty:
            for i, pset in enumerate(pred_sets):
                if len(pset)==0:
                    pset.append(self.predict(test_HVs[i]))
        return pred_sets

    def point_valued_CP(self, test_HVs, alpha, allow_empty=True):
        ''' Trim the marginal Jackknife+ prediction sets at significance level alpha to
            produce point predictions.

            Args:
                test_HVs: Input hypervectors
                alpha: significance level of the conformal psets
                allow_empty: If False, the point predictor falls back to a plain
                    prediction when the conformal pset is empty
        '''

        test_HVs = np.atleast_2d(np.asarray(test_HVs))
        counts = self._comparison_counts(test_HVs)
        mask = self._inclusion_mask(counts, alpha)

        # Trim to the most conformal label in the pset: the one fewest models rejected.
        candidate_counts = np.where(mask, counts, np.iinfo(counts.dtype).max)
        best_counts = candidate_counts.min(axis=1, keepdims=True)

        # Counts are integers, so settle any tie by similarity to the full-data prototypes.
        tie_sims = np.where(
            candidate_counts == best_counts,
            self.full_model._sim_matrix(test_HVs),
            -np.inf,
        )
        chosen = np.argmax(tie_sims, axis=1)

        pred_sets = [
            [self.class_labels[chosen[i]]] if non_empty else []
            for i, non_empty in enumerate(mask.any(axis=1))
        ]

        if not allow_empty:
            for i, pset in enumerate(pred_sets):
                if len(pset) == 0:
                    #TODO
                    pred_sets[i] = self.predict(test_HVs[i])

        return pred_sets

    def predict(self, test_HVs):
        ''' Performs standard HDC classification using prototypes built from the whole
            development set (no conformal trimming).

            Args:
                test_HVs: Input hypervectors (n_test, dim)
            Returns:
                predictions: Array of predicted class labels
        '''
        self._check_fitted()
        return self.full_model.predict(test_HVs)
