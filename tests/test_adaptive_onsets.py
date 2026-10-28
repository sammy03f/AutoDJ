"""Known local statistics, threshold-curve peak rules, and evaluation parity."""

import unittest

import numpy as np

from autodj.analysis import local_onset_threshold, pick_onsets
from evals import onset_adaptive, onset_sensitivity


class LocalThresholdTests(unittest.TestCase):
    def test_population_std_and_clipped_edges(self):
        result = local_onset_threshold([1, 3, 5], radius=1, std_multiplier=1, floor=0)
        np.testing.assert_allclose(result, [3, 3 + np.sqrt(8 / 3), 5])

    def test_floor_constant_empty_singleton_and_generator(self):
        np.testing.assert_array_equal(local_onset_threshold([0, 0, 0], radius=8), [0.1] * 3)
        np.testing.assert_array_equal(local_onset_threshold([2, 2], radius=8), [2, 2])
        self.assertEqual(local_onset_threshold([], radius=1).size, 0)
        np.testing.assert_array_equal(local_onset_threshold(iter([2]), radius=8), [2])

    def test_scaling_when_floor_is_zero_and_input_preserved(self):
        values = np.array([0., 1, 4, 2, 0])
        original = values.copy()
        result = local_onset_threshold(values, radius=2, floor=0)
        np.testing.assert_allclose(local_onset_threshold(values * 5, radius=2, floor=0), result * 5)
        np.testing.assert_array_equal(values, original)
        self.assertFalse(np.shares_memory(result, values))

    def test_quiet_isolated_peak_passes_but_silence_does_not(self):
        values = np.zeros(17)
        values[8] = 1
        thresholds = local_onset_threshold(values, radius=8)
        adaptive = pick_onsets(values, range(17), threshold=thresholds, min_interval=0)
        fixed = pick_onsets(values, range(17), threshold=10, min_interval=0)
        np.testing.assert_array_equal(adaptive.frame_indices, [8])
        self.assertEqual(fixed.times.size, 0)
        self.assertEqual(pick_onsets(np.zeros(17), range(17),
                                    threshold=local_onset_threshold(np.zeros(17), radius=8),
                                    min_interval=0).times.size, 0)

    def test_including_peak_can_suppress_it_in_short_neighborhood(self):
        values = [0, 1, 0]
        thresholds = local_onset_threshold(values, radius=8)
        self.assertGreater(thresholds[1], 1)
        self.assertEqual(pick_onsets(values, range(3), threshold=thresholds, min_interval=0).times.size, 0)

    def test_invalid_inputs(self):
        for radius in (0, -1, True, 1.5):
            with self.subTest(radius=radius), self.assertRaises((TypeError, ValueError)):
                local_onset_threshold([1], radius=radius)
        for key in ("floor", "std_multiplier"):
            for value in (-1, np.inf, np.nan):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    local_onset_threshold([1], radius=1, **{key: value})
        for values in ([-1], [np.nan], [np.inf], [[1]], [1j], [1e308, 1e308]):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                local_onset_threshold(values, radius=1)


class ThresholdCurvePickingTests(unittest.TestCase):
    def pick(self, values, threshold, min_interval=0):
        return pick_onsets(values, range(len(values)), threshold=threshold, min_interval=min_interval)

    def test_scalar_and_constant_curve_equivalence(self):
        values = [0, 2, 0, 5, 5, 0, 3, 0]
        for interval in (0, 3):
            scalar = self.pick(values, np.array(1.0), interval)
            curve = self.pick(values, iter([1] * len(values)), interval)
            np.testing.assert_array_equal(scalar.frame_indices, curve.frame_indices)

    def test_per_frame_threshold_uses_strict_comparison(self):
        result = self.pick([0, 2, 0, 5, 0, 3, 0], [0, 2, 0, 4, 0, 4, 0])
        np.testing.assert_array_equal(result.frame_indices, [3])

    def test_plateau_uses_first_threshold_only(self):
        self.assertEqual(self.pick([0, 4, 4, 0], [0, 5, 0, 0]).times.size, 0)
        np.testing.assert_array_equal(self.pick([0, 4, 4, 0], [0, 3, 5, 0]).frame_indices, [1])

    def test_invalid_curve_and_invalid_scalar_on_empty(self):
        for threshold in ([0], [[0, 1, 0]], [0, np.nan, 0], [0, -1, 0], [0, 1j, 0]):
            with self.subTest(threshold=threshold), self.assertRaises((ValueError, TypeError)):
                self.pick([0, 2, 0], threshold)
        for threshold in (np.inf, np.nan, -1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.pick([], threshold)


class AdaptiveEvaluationTests(unittest.TestCase):
    def test_seed_groups_are_disjoint(self):
        self.assertFalse(set(onset_adaptive.SEED_GROUPS["baseline"]) &
                         set(onset_adaptive.SEED_GROUPS["fresh"]))

    def test_fixed_arm_reproduces_existing_scoring_and_samples(self):
        for args in (("sharp_burst", 1, 0, 7, 0), ("click", 0.1, 0.2, 19, 31),
                     (None, 0, 0.2, 43, 0)):
            with self.subTest(args=args):
                old = onset_sensitivity.evaluate_case(*args)
                new = onset_adaptive.evaluate_case(*args)
                self.assertEqual(new["input_sha256"], old["input_sha256"])
                self.assertEqual(new["metrics"]["fixed"], old["metrics"]["frame_center"])
                self.assertEqual(new["times"]["fixed"], old["times"]["frame_center"])

    def test_summary_keeps_misses_and_both_detectors(self):
        case = onset_adaptive.evaluate_case("sharp_burst", 0.1, 0, 101, 0)
        summary = onset_adaptive.summarize([case])
        self.assertEqual(summary["methods"]["fixed"]["missed_count"], 1)
        self.assertEqual(summary["methods"]["adaptive"]["matched_count"], 1)
        self.assertIsNone(summary["methods"]["fixed"]["mean_absolute_error_seconds"])
