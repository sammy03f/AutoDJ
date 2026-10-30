"""Known raw MAD values, zero-spread behavior, and baseline compatibility."""

import unittest

import numpy as np

from autodj.analysis import robust_onset_threshold, local_onset_threshold, pick_onsets
from evals import onset_robust as robust
from evals import onset_sequences as sequences
from evals import onset_sensitivity as isolated


class RobustThresholdTests(unittest.TestCase):
    def test_known_median_and_raw_mad_without_scale_factor(self):
        # Median=3, deviations [2,1,0,1,97], MAD=1, threshold=7.
        result = robust_onset_threshold([1, 2, 3, 4, 100], radius=2, floor=0)
        self.assertEqual(result[2], 7)

    def test_clipped_even_neighborhoods_and_no_padding(self):
        np.testing.assert_allclose(robust_onset_threshold([1, 3, 5], radius=1, floor=0), [6, 11, 8])

    def test_empty_singleton_constant_and_floor(self):
        self.assertEqual(robust_onset_threshold([], radius=2).size, 0)
        np.testing.assert_array_equal(robust_onset_threshold([0], radius=12), [0.1])
        np.testing.assert_array_equal(robust_onset_threshold([2, 2, 2], radius=12), [2, 2, 2])
        np.testing.assert_array_equal(robust_onset_threshold([1, 3, 5], radius=1, mad_multiplier=0, floor=0), [2, 3, 4])

    def test_large_peak_does_not_raise_sparse_background_threshold(self):
        values = [1, 1, 1, 100, 1, 1, 1]
        curve = robust_onset_threshold(values, radius=3)
        self.assertEqual(curve[3], 1)
        self.assertGreater(local_onset_threshold(values, radius=3)[3], 100)
        result = pick_onsets(values, range(7), threshold=curve, min_interval=0)
        np.testing.assert_array_equal(result.frame_indices, [3])

    def test_zero_mad_can_admit_small_background_fluctuations(self):
        values = [1, 1, 1, 1.001, 1, 1, 1]
        result = pick_onsets(values, range(7), threshold=robust_onset_threshold(values, radius=3), min_interval=0)
        np.testing.assert_array_equal(result.frame_indices, [3])

    def test_scaling_generator_and_no_mutation(self):
        values = np.array([0., 1, 3, 8, 2])
        original = values.copy()
        curve = robust_onset_threshold(iter(values), radius=2, floor=0)
        np.testing.assert_allclose(robust_onset_threshold(5 * values, radius=2, floor=0), curve * 5)
        np.testing.assert_array_equal(values, original)
        self.assertFalse(np.shares_memory(values, curve))

    def test_invalid_parameters_and_input(self):
        for radius in (0, -1, 1.5, True):
            with self.subTest(radius=radius), self.assertRaises((ValueError, TypeError)):
                robust_onset_threshold([0], radius=radius)
        for name in ("floor", "mad_multiplier"):
            for value in (-1, np.nan, np.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    robust_onset_threshold([0], radius=1, **{name: value})
        for values in ([-1], [np.nan], [np.inf], [[1]], [1j], [1e308, 1e308]):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                robust_onset_threshold(values, radius=1)


class RobustEvaluationTests(unittest.TestCase):
    def test_clean_dense_bursts_recover_with_robust_threshold(self):
        samples, starts, _ = sequences.make_sequence("sharp_burst", 120, 4, "steady", 1, 0, 1013)
        result = robust.evaluate_signal(samples, (starts / isolated.RATE).tolist(), {})
        self.assertEqual(result["metrics"]["selected_adaptive"]["matched_count"], 2)
        self.assertEqual(result["metrics"]["median_mad"]["matched_count"], 8)
        self.assertEqual(result["metrics"]["median_mad"]["false_count"], 0)

    def test_sequence_baselines_unchanged(self):
        args = ("sharp_burst", 120, 4, "alternating", 0.1, 0.05, 701)
        samples, starts, _ = sequences.make_sequence(*args)
        new = robust.evaluate_signal(samples, (starts / isolated.RATE).tolist(), {})
        old = sequences.evaluate_case(*args)
        self.assertEqual(new["input_sha256"], old["input_sha256"])
        for method in sequences.METHODS:
            self.assertEqual(new["metrics"][method], old["metrics"][method])
            self.assertEqual(new["frame_indices"][method], old["frame_indices"][method])

    def test_isolated_fixed_baseline_preserved(self):
        args = ("click", 0.1, 0.2, 401, 31)
        samples, _ = isolated.make_noisy_signal(*args)
        case = robust.evaluate_signal(samples, [(512 + 31) / isolated.RATE], {})
        original = isolated.evaluate_case(*args)
        for key in ("event_count", "matched_count", "missed_count", "candidate_count", "false_count"):
            self.assertEqual(case["metrics"]["fixed"][key], original["metrics"]["frame_center"][key])

    def test_silence_controls_have_no_candidates(self):
        case = robust.evaluate_signal(np.zeros(1536), [], {})
        for method in robust.METHODS:
            self.assertEqual(case["metrics"][method]["candidate_count"], 0)
        summary = robust.summarize([case])
        self.assertIsNone(summary["methods"]["median_mad"]["recall"])

    def test_fresh_noise_seeds_are_disjoint_from_previous_studies(self):
        previous = {7, 19, 43, 101, 211, 307, 401, 503, 607, 701, 809}
        for name in ("isolated_fresh", "sequences_fresh"):
            self.assertFalse(set(robust.SEED_GROUPS[name]) & previous)

    def test_unknown_regime_is_rejected(self):
        with self.assertRaises(ValueError):
            robust.evaluate_group("unknown", (1013,))
