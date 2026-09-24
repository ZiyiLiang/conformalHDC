import numpy as np
from scipy.spatial.distance import cdist


def sets_from_mask(mask, labels):
    ''' One prediction set per row of a boolean (n, n_class) mask, in labels order. '''
    return [[label for label, keep in zip(labels, row) if keep] for row in mask]


class ConformalHDC():
    def __init__(self, class_HVs, class_labels, sim_measure="cosine",
                random_state=0, verbose=True):

        self.class_HVs = class_HVs   # shape (n_class, dim)
        self.class_labels = class_labels  # real labels (e.g. [1, 5, 9])
        # Canonical indices (0, 1, 2...) for internal array indexing
        self.canonical_indices = np.arange(len(self.class_HVs))
        # Map real labels to canonical indices
        self.label_to_idx = {label: i for i, label in enumerate(class_labels)}

        self.sim_measure = sim_measure

        self.random_state = random_state
        self.verbose = verbose

    def compute_calib_scores(self, calib_HVs, calib_labels, score_type="discount", **kwargs):
        # Convert incoming real labels to canonical indices
        # If calib_labels are tensors, convert to list/numpy first
        if hasattr(calib_labels, 'tolist'):
            calib_labels = calib_labels.tolist()
        canonical_labels = np.array([self.label_to_idx[l] for l in calib_labels], dtype=int)

        # Pass canonical labels to the internal score function
        self.calib_scores = self._compute_nonconformity_scores(
                    calib_HVs, canonical_labels, score_type=score_type, **kwargs
                )
        self.n_calib = len(self.calib_scores)
        self.score_type = score_type

        self.calib_scores_per_label = [self.calib_scores[canonical_labels == c]
                                       for c in self.canonical_indices]
        self.n_calib_per_label = [len(scores) for scores in self.calib_scores_per_label]


    def _sim_matrix(self, HVs):
        ''' Similarities between every HV and every class prototype.

            Args:
                HVs: Input hypervectors, shape (n, dim). A single (dim,) HV is also accepted.
            Returns:
                sim_matrix: shape (n, n_class), entry [i, c] is the similarity of HVs[i] and class_HVs[c]
        '''
        HVs = np.atleast_2d(np.asarray(HVs))

        prototypes = np.asarray(self.class_HVs)[self.canonical_indices]
        dtype = np.result_type(HVs.dtype, prototypes.dtype, np.float32)
        HVs = HVs.astype(dtype, copy=False)
        prototypes = prototypes.astype(dtype, copy=False)

        if self.sim_measure == "euclidean":
            # cdist avoids an (n, n_class, dim) temporary and cancellation
            # from expanding squared distances into dot products.
            distances = cdist(HVs.real, prototypes.real)
            if np.iscomplexobj(HVs) or np.iscomplexobj(prototypes):
                distances = np.hypot(
                    distances, cdist(HVs.imag, prototypes.imag),
                )
            sim_matrix = np.full(distances.shape, 1e8)
            np.divide(1.0, distances, out=sim_matrix, where=distances != 0)

        elif self.sim_measure == "cosine":
            dots = HVs @ prototypes.T
            norms = (np.linalg.norm(HVs, axis=1)[:, None]
                     * np.linalg.norm(prototypes, axis=1)[None, :])
            sim_matrix = (1 + dots / (norms + 1e-8)) / 2

        elif self.sim_measure == "complex_cosine":
            dots = (HVs @ prototypes.conj().T).real
            sim_matrix = (1 + dots / HVs.shape[1]) / 2

        else:
            raise ValueError(f"Unknown similarity measure: {self.sim_measure}")

        return np.asarray(sim_matrix, dtype=float)


    def _uniform_draws(self, n):
        ''' Seeded uniforms that randomize inverse-quantile scores for exact coverage.
            Every call reuses the same draws, so row i always receives the same U.
        '''
        return np.random.RandomState(self.random_state).uniform(0, 1, size=n)


    @staticmethod
    def _softmax_cumulative(sim_matrix):
        ''' Softmax probabilities of the similarities and, for every class, the probability
            mass accumulated up to that class when a row is sorted ascending.
        '''
        exp_sims = np.exp(sim_matrix - sim_matrix.max(axis=1, keepdims=True))
        pi_hat = exp_sims / exp_sims.sum(axis=1, keepdims=True)

        indices_sorted = np.argsort(pi_hat, axis=1)
        cumulative_probs = np.empty_like(pi_hat)
        np.put_along_axis(
            cumulative_probs, indices_sorted,
            np.take_along_axis(pi_hat, indices_sorted, axis=1).cumsum(axis=1), axis=1,
        )
        return pi_hat, cumulative_probs


    def _inverse_quantile_scores(self, sim_matrix, targets, U=None):
        ''' Generalized Inverse Quantile Score.
            Turns similarities into probabilities via Softmax, accumulates probability
            mass up to the target label, and randomizes to get exact coverage.

            U: one uniform draw per row. Supplied by callers that batch several models
               into one matrix and need the same draw reused per sample.
        '''
        rows = np.arange(len(targets))
        pi_hat, cumulative_probs = self._softmax_cumulative(sim_matrix)
        if U is None:
            U = self._uniform_draws(len(targets))
        return -cumulative_probs[rows, targets] + U * pi_hat[rows, targets]


    @staticmethod
    def _score_formula(sim_true_class, sim_all_classes, score_type, **kwargs):
        ''' Similarity-based nonconformity scores; the inputs broadcast elementwise. '''
        sim_other_classes = sim_all_classes - sim_true_class

        if score_type == "discount":
            return -(sim_true_class / sim_all_classes) * sim_true_class
        elif score_type == "alt_discount":
            return -(sim_true_class / sim_other_classes) * sim_true_class
        elif score_type == "ratio":
            return -sim_true_class / sim_all_classes
        elif score_type == "alt_ratio":
            return -sim_true_class / sim_other_classes
        elif score_type == "sim":
            return -sim_true_class
        elif score_type == "penalized":
            penalty = kwargs.get("penalty", 1)
            return -sim_true_class + penalty * sim_other_classes
        else:
            raise ValueError(f"Unknown score type: {score_type}")


    def _scores_from_sims(self, sim_matrix, canonical_targets,
                          score_type="discount", U=None, **kwargs):
        ''' Converts a (n, n_class) similarity matrix into nonconformity scores.
            Larger scores mean the target label fits its HV worse.

            Args:
                sim_matrix: similarities against every class, shape (n, n_class)
                canonical_targets: canonical index of the target label per row
                score_type: which nonconformity score to use
        '''
        targets = np.asarray(canonical_targets).astype(int)
        sim_matrix = np.asarray(sim_matrix, dtype=float)[:len(targets)]

        if score_type == "inverse_quantile":
            return self._inverse_quantile_scores(sim_matrix, targets, U=U)
        return self._score_formula(
            sim_matrix[np.arange(len(targets)), targets], sim_matrix.sum(axis=1),
            score_type, **kwargs,
        )


    def _class_scores(self, sim_matrix, score_type="discount", U=None, **kwargs):
        ''' Scores of every row against every candidate class, shape (n, n_class).
            Column c equals _scores_from_sims(sim_matrix, [c] * n) exactly,
            including the seeded draws shared by every column.
        '''
        sim_matrix = np.asarray(sim_matrix, dtype=float)

        if score_type == "inverse_quantile":
            pi_hat, cumulative_probs = self._softmax_cumulative(sim_matrix)
            if U is None:
                U = self._uniform_draws(len(sim_matrix))
            return -cumulative_probs + np.asarray(U)[:, None] * pi_hat
        return self._score_formula(
            sim_matrix, sim_matrix.sum(axis=1, keepdims=True), score_type, **kwargs,
        )


    def _column_update_scores(self, sim_matrix, canonical_targets, c, new_column,
                              score_type="discount", U=None, **kwargs):
        ''' Scores of each row against its target after similarity column c is replaced.

            Scores m models at once that differ from sim_matrix only in column c:
            new_column has shape (n, m), and column a of the (n, m) result equals
            _scores_from_sims on the matrix with column c set to new_column[:, a],
            up to rounding (inverse_quantile ranks tied probabilities above the target).
        '''
        sims = np.asarray(sim_matrix, dtype=float)
        targets = np.asarray(canonical_targets).astype(int)
        rows = np.arange(len(targets))
        new_column = np.asarray(new_column, dtype=float)
        is_c = (targets == c)[:, None]

        if score_type != "inverse_quantile":
            sim_true_class = np.where(is_c, new_column, sims[rows, targets][:, None])
            sim_all_classes = sims.sum(axis=1)[:, None] - sims[:, [c]] + new_column
            return self._score_formula(sim_true_class, sim_all_classes, score_type, **kwargs)

        # Unnormalized softmax mass sorted at or below the target, updated without re-sorting.
        exps, new_exps = np.exp(sims), np.exp(new_column)
        exps[:, c] = 0  # column c enters through new_exps only
        target_exps = np.where(is_c, new_exps, exps[rows, targets][:, None])
        below = (np.where(exps < exps[rows, targets][:, None], exps, 0).sum(axis=1)[:, None]
                 + np.where(new_exps < target_exps, new_exps, 0))
        rows_c = is_c[:, 0]
        others = exps[rows_c][:, :, None]
        below[rows_c] = (others * (others < new_exps[rows_c][:, None, :])).sum(axis=1)

        if U is None:
            U = self._uniform_draws(len(targets))
        total = exps.sum(axis=1)[:, None] + new_exps
        return (-(below + target_exps) + np.asarray(U)[:, None] * target_exps) / total


    def _compute_nonconformity_scores(self, HVs, canonical_targets,
                                   score_type="discount", **kwargs):
        ''' Computes the nonconformity scores of the HVs
        '''
        return self._scores_from_sims(
            self._sim_matrix(HVs), canonical_targets,
            score_type=score_type, **kwargs,
        )


    def _is_calibrated(self):
        required_attrs = ['calib_scores', 'calib_scores_per_label', 'score_type']
        if all(hasattr(self, attr) for attr in required_attrs):
            return True
        print("Compute calibration scores first! (Call function compute_calib_scores)")
        return False


    def _thresholds(self, alpha, marginal):
        ''' Score threshold of every class at significance level alpha. '''
        if marginal:
            self.quantile = np.quantile(self.calib_scores, (self.n_calib+1)*(1-alpha)/self.n_calib)
            return np.full(len(self.canonical_indices), self.quantile)
        self.quantiles = [np.quantile(scores, (n+1)*(1-alpha)/n) for scores, n in zip(self.calib_scores_per_label, self.n_calib_per_label)]
        return np.array(self.quantiles)


    def _test_scores(self, test_HVs, **kwargs):
        ''' Nonconformity scores of every test HV against every class, shape (n_test, n_class). '''
        return self._class_scores(self._sim_matrix(test_HVs), self.score_type, **kwargs)


    def set_valued_CP(self, test_HVs, alpha,
                    allow_empty=True,
                    marginal=False, **kwargs):
        ''' Computes the conformal prediction sets at significance level alpha
        '''
        if not self._is_calibrated():
            return

        keep = self._test_scores(test_HVs, **kwargs) <= self._thresholds(alpha, marginal)
        psets = sets_from_mask(keep, self.class_labels)
        if not allow_empty:
            for i, pset in enumerate(psets):
                if len(pset)==0:
                    pset.append(self.predict(test_HVs[i]))

        return psets


    def point_valued_CP(self, test_HVs, alpha,
                    allow_empty=True,
                    marginal=False, **kwargs):
        ''' Trim the conformal prediction sets at significance level alpha to produce point prediction.

            Args:
                test_HVs: Input hypervectors
                alpha: significance level of the conformal psets
                allow_empty: If False, the point predictor will use vanilla HDC prediction when the
                    conformal pset is empty
                marginal: If True, compute marginal psets, otherwise, compute label-conditional psets
        '''
        if not self._is_calibrated():
            return

        keep = self._test_scores(test_HVs, **kwargs) <= self._thresholds(alpha, marginal)
        sims = self._sim_matrix(test_HVs) #(n, nc)

        # Trim each pset to its most similar class; argmax keeps the first of tied classes.
        best_in_set = np.argmax(np.where(keep, sims, -np.inf), axis=1)
        # if empty pset, choose ordinary HDC predictions
        best_overall = np.argmax(sims, axis=1)
        return [
            [self.class_labels[in_set]] if row.any()
            else ([] if allow_empty else [self.class_labels[overall]])
            for row, in_set, overall in zip(keep, best_in_set, best_overall)
        ]


    def get_max_p_value(self, test_HVs, marginal=False, **kwargs):
        ''' Calculates the p-value based OOD score (credibility) for each test sample.

            Args:
                test_HVs: Test hypervectors or features.
                marginal (bool):
                    If False (default): Computes label-conditional p-values.
                    If True: Computes marginal p-values.

            Returns:
                array of shape (n_test,) with values in [0, 1].
                High score = Inlier (fits distribution well).
                Low score  = Outlier (fits no classes well).
        '''
        if not self._is_calibrated():
            return

        # Nonconformity score of the test point against every class: [n_test, n_classes]
        test_scores_matrix = self._test_scores(test_HVs, **kwargs)

        # Calculate p-values based OOD scores
        if marginal:
            all_calib = np.sort(np.ravel(self.calib_scores))
            n_calib = len(all_calib)

            # Since we assume High Score = Outlier, the "predicted" class is the one with the Lowest nonconformity score (best fit).
            min_test_scores = np.min(test_scores_matrix, axis=1)

            # Calculate p-value against the all calibration points
            ranks = np.searchsorted(all_calib, min_test_scores, side='left')
            return (n_calib - ranks + 1) / (n_calib + 1)

        p_values_matrix = np.zeros(test_scores_matrix.shape)
        for c_idx in self.canonical_indices:
            calib_c = np.sort(self.calib_scores_per_label[c_idx])
            n_calib = len(calib_c)
            if n_calib == 0:
                continue
            # Compute label-conditional p-value within this specific class
            ranks = np.searchsorted(calib_c, test_scores_matrix[:, c_idx], side='left')
            p_values_matrix[:, c_idx] = (n_calib - ranks + 1) / (n_calib + 1)

        # Return the best p-value across all possible classes
        return np.max(p_values_matrix, axis=1)

    def predict(self, test_HVs):
        test_HVs = np.asarray(test_HVs)

        if test_HVs.ndim == 1:
            test_HVs = test_HVs.reshape(1, -1)
        elif test_HVs.ndim != 2:
            raise ValueError(
                f"Input must be a 2D array. Got shape {test_HVs.shape}"
            )

        sims = self._sim_matrix(test_HVs)
        best_indices = np.argmax(sims, axis=1)
        return [self.class_labels[idx] for idx in best_indices]


class CachedConformalHDC(ConformalHDC):
    """Cache fixed-model similarities for one repetition.

    Registered arrays and class prototypes must remain unchanged until the
    cache is replaced. Retain array references so identity checks stay valid.
    """

    def cache_similarities(self, *arrays):
        compute = super()._sim_matrix
        self._cached_sims = [(x, compute(x)) for x in arrays]
        return self

    def _sim_matrix(self, HVs):
        for original, sims in getattr(self, "_cached_sims", ()):
            if HVs is original:
                return sims
        return super()._sim_matrix(HVs)
