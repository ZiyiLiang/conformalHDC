"""Unit checks for score definitions and leakage-free operating thresholds."""
import unittest
import numpy as np

from experiments_ood.common import (
    EXPERIMENTS, apply_quantizer, fit_quantizer, lower_tail_threshold,
    ood_scores, split_indices,
)


class OODProtocolTests(unittest.TestCase):
    def test_score_definitions_share_cosine_matrix(self):
        cosine = np.array([[0.8, 0.2], [-0.2, 0.4]])
        scores = ood_scores(cosine)
        delta = (1 + cosine) / 2
        np.testing.assert_allclose(scores["maxsim"], cosine.max(1))
        np.testing.assert_allclose(scores["discount"], delta.max(1) ** 2 / delta.sum(1))
        np.testing.assert_allclose(scores["energy"], np.log(np.exp(cosine).sum(1)))

    def test_energy_is_stable_at_small_temperature(self):
        scores = ood_scores(np.array([[1.0, -1.0]]), temperature=1e-5)
        self.assertTrue(np.isfinite(scores["energy"]).all())
        self.assertAlmostEqual(scores["energy"][0], 1.0)

    def test_lower_tail_cutoff_has_finite_sample_rank(self):
        values = np.array([1., 2., 3., 4.])
        rejected = 0
        for i, test in enumerate(values):
            threshold = lower_tail_threshold(np.delete(values, i), 0.25)
            rejected += test < threshold
        self.assertEqual(rejected, 1)

    def test_ties_are_accepted_and_small_alpha_rejects_nothing(self):
        threshold = lower_tail_threshold(np.array([1., 1., 2.]), 0.5)
        self.assertEqual(threshold, 1.0)
        self.assertFalse(1.0 < threshold)
        self.assertEqual(lower_tail_threshold(np.array([1., 1., 2.]), 0.01), -np.inf)

    def test_held_out_features_cannot_change_fitted_quantization(self):
        minimum, span = fit_quantizer(np.array([[1.], [3.]]))
        np.testing.assert_array_equal(apply_quantizer(np.array([[-99.], [99.]]), minimum, span), [[0], [20]])
        np.testing.assert_array_equal(minimum, [1.])
        np.testing.assert_array_equal(span, [2.])

    def test_image_split_preserves_every_declared_ood_example(self):
        y = np.repeat(np.arange(10), np.arange(20, 30))
        train, cal, test, ood = split_indices(y, EXPERIMENTS["mnist"], 1)
        np.testing.assert_array_equal(ood, np.flatnonzero(y >= 6))
        for left, right in [(train, cal), (train, test), (cal, test), (train, ood)]:
            self.assertEqual(len(np.intersect1d(left, right)), 0)


if __name__ == "__main__":
    unittest.main()
