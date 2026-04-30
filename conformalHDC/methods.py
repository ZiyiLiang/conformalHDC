import numpy as np


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
        self.calib_scores_per_label = [[] for _ in self.canonical_indices]
        self.n_calib_per_label = [0] * len(self.canonical_indices)

        # Convert incoming real labels to canonical indices
        # If calib_labels are tensors, convert to list/numpy first
        if hasattr(calib_labels, 'tolist'):
            calib_labels = calib_labels.tolist()
        canonical_labels = [self.label_to_idx[l] for l in calib_labels]

        # Pass canonical labels to the internal score function
        self.calib_scores = self._compute_nonconformity_scores(
                    calib_HVs, canonical_labels, score_type=score_type, **kwargs
                )        
        self.n_calib = len(self.calib_scores)
        self.score_type = score_type

        for i, c_idx in enumerate(canonical_labels):
            self.calib_scores_per_label[c_idx].append(self.calib_scores[i])
            self.n_calib_per_label[c_idx] += 1

    
    def _sim(self, HV1s, HV2s):
        ''' Measures the similarities between HVs, larger values corresponds to more similar HVs.
            Automatically handles 1D vectors by reshaping them to (1, D).
            Supports broadcasting (e.g., comparing N vectors vs 1 vector).
        '''
        # Ensure inputs are at least 2D (shape [N, D] or [1, D])
        HV1s = np.atleast_2d(HV1s) 
        HV2s = np.atleast_2d(HV2s)

        if self.sim_measure == "euclidean":
            # sims = 1/np.linalg.norm(HV1s - HV2s, axis=1)
            norm = np.linalg.norm(HV1s - HV2s, axis=1)
            sims = np.where(norm == 0, 1e8, 1.0 / norm)

        elif self.sim_measure == "cosine":
            dot_products = (HV1s * HV2s).sum(axis=1)
            an = np.linalg.norm(HV1s, axis=1)
            bn = np.linalg.norm(HV2s, axis=1)
            epsilon = 1e-8
            # sims = dot_products / (an * bn + epsilon)  # range from [-1,1] 
            sims = dot_products / (an * bn + epsilon) 
            sims = (1 + sims) / 2  # ranges from [0,1]

        elif self.sim_measure == "complex_cosine":
            D = HV1s.shape[1]
            # sims = np.real(np.sum(HV1s * np.conj(HV2s), axis=1)) / D # range from [-1,1] 
            sims = np.real(np.sum(HV1s * np.conj(HV2s), axis=1)) / D 
            sims = (1 + sims) / 2  # ranges from [0,1]
        
        else:
            print("Unknown similarity measures!")
        
        return sims


    def _compute_nonconformity_scores(self, HVs, canonical_targets,
                                   score_type="discount", **kwargs):
        ''' Computes the nonconformity scores of the HVs
        '''
        rng = np.random.RandomState(self.random_state)
        scores = np.zeros(len(canonical_targets))

        for i, (HV, target_idx) in enumerate(zip(HVs, canonical_targets)):
            sims_list = []
            sim_all_classes = 0
            sim_true_class = 0
            for c_idx in self.canonical_indices:
                sim = self._sim(HV.reshape(1, -1), self.class_HVs[c_idx].reshape(1, -1))
                sims_list.append(sim.item()) # Ensure it's a scalar

                sim_all_classes += sim
                if c_idx == target_idx:
                    sim_true_class = sim

            if score_type == "discount":
                score = -(sim_true_class/sim_all_classes)*(sim_true_class)
            elif score_type == "alt_discount":
                score = -(sim_true_class/(sim_all_classes-sim_true_class))*(sim_true_class)
            elif score_type == "ratio":
                score = -sim_true_class/sim_all_classes
            elif score_type == "alt_ratio":
                score = -sim_true_class/(sim_all_classes-sim_true_class)
            elif score_type == "sim":
                score = -sim_true_class
            elif score_type == "penalized":
                penalty = kwargs.get("penalty",1)
                sim_other_classes = sim_all_classes - sim_true_class
                score = -sim_true_class + penalty * sim_other_classes
            # Generalized Inverse Quantile Score 
            elif score_type == "inverse_quantile":
                # Convert similarities to probabilities via Softmax
                sims_arr = np.array(sims_list)
                exp_sims = np.exp(sims_arr - np.max(sims_arr))
                pi_hat = exp_sims / np.sum(exp_sims)
                
                pi_true = pi_hat[target_idx]
                indices_sorted = np.argsort(pi_hat)
                pi_sorted = pi_hat[indices_sorted]

                rank_idx = np.where(indices_sorted == target_idx)[0][0]
                cumulative_prob = np.sum(pi_sorted[:rank_idx+1])
                
                # randomize to get exact coverage
                U = rng.uniform(0, 1)
                score = - cumulative_prob + U * pi_true
            else:
                 print("Unknown score type!")
            scores[i] = score 
            
        return scores
    

    def set_valued_CP(self, test_HVs, alpha, 
                    allow_empty=True,
                    marginal=False, **kwargs):
        ''' Computes the conformal prediction sets at significance level alpha
        '''

        n_test = len(test_HVs)
        psets = [[] for _ in range(n_test)]

        # check if calibration scores are computed
        required_attrs = ['calib_scores', 'calib_scores_per_label', 'score_type']
        if not all(hasattr(self, attr) for attr in required_attrs):
            print("Compute calibration scores first! (Call function compute_calib_scores)")
            return 
        
        if marginal: 
            self.quantile = np.quantile(self.calib_scores, (self.n_calib+1)*(1-alpha)/self.n_calib)
        else:
            self.quantiles = [np.quantile(scores, (n+1)*(1-alpha)/n) for scores, n in zip(self.calib_scores_per_label, self.n_calib_per_label)]
        
        for c_idx in self.canonical_indices:
            test_scores = self._compute_nonconformity_scores(
                test_HVs, [c_idx]*n_test, score_type=self.score_type, **kwargs
            )
            for i in range(n_test):
                q = self.quantile if marginal else self.quantiles[c_idx]
                if test_scores[i] <= q:
                    real_label = self.class_labels[c_idx]
                    psets[i].append(real_label)
            
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

        n_test = len(test_HVs)
        psets = [[] for _ in range(n_test)]

        # check if calibration scores are computed
        required_attrs = ['calib_scores', 'calib_scores_per_label', 'score_type']
        if not all(hasattr(self, attr) for attr in required_attrs):
            print("Compute calibration scores first! (Call function compute_calib_scores)")
            return 
        
        if marginal: 
            self.quantile = np.quantile(self.calib_scores, (self.n_calib+1)*(1-alpha)/self.n_calib)
        else:
            self.quantiles = [np.quantile(scores, (n+1)*(1-alpha)/n) for scores, n in zip(self.calib_scores_per_label, self.n_calib_per_label)]
        
        for c_idx in self.canonical_indices:
            test_scores = self._compute_nonconformity_scores(
                test_HVs, [c_idx]*n_test, score_type=self.score_type, **kwargs
            )
            for i in range(n_test):
                q = self.quantile if marginal else self.quantiles[c_idx]
                if test_scores[i] <= q:
                    psets[i].append(c_idx)
        
        # Trim the psets and convert to real labels
        for i, pset in enumerate(psets):
            if len(pset) == 1:
                psets[i] = [self.class_labels[pset[0]]]
            elif len(pset) > 1:
                cand_prototypes = self.class_HVs[pset]
                query_vec = test_HVs[i].reshape(1, -1)
                query_broadcasted = np.tile(query_vec, (len(pset), 1))
                pset_sims = self._sim(query_broadcasted, cand_prototypes)
                psets[i] = [self.class_labels[pset[np.argmax(pset_sims)]]]
                # psets[i] = self.predict(test_HVs[i])
            elif len(pset) == 0 and not allow_empty:
                psets[i] = self.predict(test_HVs[i])

        return psets


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
        n_test = len(test_HVs)
        n_classes = len(self.canonical_indices)
        
        # Check prerequisites
        required_attrs = ['calib_scores', 'calib_scores_per_label', 'score_type']
        if not all(hasattr(self, attr) for attr in required_attrs):
            print("Compute calibration scores first! (Call function compute_calib_scores)")
            return 
        
        # Compute the nonconformity score of the test point against every class: [n_test, n_classes]
        test_scores_matrix = np.zeros((n_test, n_classes))
        for i, c_idx in enumerate(self.canonical_indices):
            test_scores_matrix[:, i] = self._compute_nonconformity_scores(
                test_HVs, [c_idx]*n_test, score_type=self.score_type, **kwargs
            )

        # Calculate p-values based OOD scores
        if marginal:
            if isinstance(self.calib_scores, list):
                all_calib = np.sort(np.concatenate(self.calib_scores))
            else:
                all_calib = np.sort(self.calib_scores.ravel())
                
            n_calib = len(all_calib)
            
            # Since we assume High Score = Outlier, the "predicted" class is the one with the Lowest nonconformity score (best fit).
            min_test_scores = np.min(test_scores_matrix, axis=1)
            
            # Calculate p-value against the all calibration points 
            ranks = np.searchsorted(all_calib, min_test_scores, side='left')
            p_values = (n_calib - ranks + 1) / (n_calib + 1)
            return p_values

        else:
            p_values_matrix = np.zeros((n_test, n_classes))
            
            for i, c_idx in enumerate(self.canonical_indices):
                calib_c = np.sort(self.calib_scores_per_label[c_idx])
                n_calib = len(calib_c)
                
                if n_calib == 0:
                    p_values_matrix[:, i] = 0.0
                    continue
                
                # Retrieve pre-calculated scores for this class
                test_scores_c = test_scores_matrix[:, i]
                
                # Compute label-conditional p-value within this specific class
                ranks = np.searchsorted(calib_c, test_scores_c, side='left')            
                p_values_matrix[:, i] = (n_calib - ranks + 1) / (n_calib + 1)
                
            # Return the best p-value across all possible classes
            return np.max(p_values_matrix, axis=1)


    def predict(self, test_HVs):
        ''' Performs standard HDC classification (Argmax Similarity).
            
            Args:
                test_HVs: Input hypervectors (n_test, dim)
            Returns:
                predictions: Array of predicted class labels
        '''
        test_HVs = np.array(test_HVs)

        # Format Check: Ensure 2D (Handle single sample case automatically)
        if test_HVs.ndim == 1:
            # Reshape (dim,) -> (1, dim)
            test_HVs = test_HVs.reshape(1, -1)
        elif test_HVs.ndim != 2:
            raise ValueError(f"Input must be a 2D array. Got shape {test_HVs.shape}")

        n_test = len(test_HVs)
        n_classes = len(self.canonical_indices)
        
        # Matrix to store similarities: [n_test, n_classes]
        sim_matrix = np.zeros((n_test, n_classes))
        
        for c_idx in self.canonical_indices:
            # Get the prototype for class c_idx 
            proto = self.class_HVs[c_idx].reshape(1, -1)
            
            # Compute similarity between ALL test HVs and THIS prototype
            sim_matrix[:, c_idx] = self._sim(test_HVs, proto)
            
        best_canonical_indices = np.argmax(sim_matrix, axis=1)        
        
        predictions = [self.class_labels[idx] for idx in best_canonical_indices]
        
        return predictions
    