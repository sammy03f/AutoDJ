"""Known periodic curves and an independent full-correlation reference."""

import unittest

import numpy as np

from autodj.analysis import estimate_tempo


class TempoTests(unittest.TestCase):
    def test_exact_periodic_impulses(self):
        strength = np.zeros(640)
        strength[::32] = 1
        result = estimate_tempo(strength, 64)
        self.assertEqual(result.bpm, 120)
        self.assertEqual(result.lag_frames, 32)
        self.assertGreater(result.score, 0.9)

    def test_curve_matches_numpy_full_correlation(self):
        values = np.array([0., 1, 4, 0, 2, 0, 1, 3, 0, 2, 1, 0])
        result = estimate_tempo(values, 10, min_bpm=100, max_bpm=300)
        centered = values - values.mean()
        reference = np.correlate(centered, centered, mode="full")[len(values) - 1:]
        np.testing.assert_allclose(result.scores, reference[result.lags] / reference[0], atol=1e-15)
        np.testing.assert_array_equal(result.lags, [2, 3, 4, 5, 6])

    def test_gain_offset_generator_and_input_preservation(self):
        values = np.zeros(640)
        values[::32] = 1
        original = values.copy()
        result = estimate_tempo(iter(values), 64)
        for modified in (values * 0.01, values * 1e300, values + 2):
            other = estimate_tempo(modified, 64)
            self.assertEqual(other.bpm, result.bpm)
            np.testing.assert_allclose(other.scores, result.scores, atol=1e-14)
        np.testing.assert_array_equal(values, original)
        self.assertFalse(np.shares_memory(values, result.scores))

    def test_silence_constant_empty_and_short_input_abstain(self):
        for values in ([], [0], [1], [0] * 128, [3] * 128, [0, 1, 0]):
            with self.subTest(length=len(values)):
                result = estimate_tempo(values, 64)
                self.assertIsNone(result.bpm)
                self.assertIsNone(result.score)
                self.assertIsNone(result.lag_frames)

    def test_two_period_support_and_inclusive_bounds(self):
        result = estimate_tempo(np.arange(20), 10, min_bpm=60, max_bpm=120)
        np.testing.assert_array_equal(result.lags, [5, 6, 7, 8, 9, 10])
        short = estimate_tempo(np.arange(9), 10, min_bpm=60, max_bpm=120)
        self.assertEqual(short.lags.size, 0)

    def test_no_positive_correlation_abstains(self):
        result = estimate_tempo([0, 1], 1, min_bpm=59, max_bpm=61)
        np.testing.assert_allclose(result.scores, [-0.5])
        self.assertIsNone(result.bpm)

    def test_alternating_strength_can_choose_half_time(self):
        values = np.zeros(960)
        values[::32] = 1
        values[32::64] = 0.1
        result = estimate_tempo(values, 64, min_bpm=50, max_bpm=200)
        self.assertEqual(result.bpm, 60)

    def test_exact_score_ties_choose_shorter_lag(self):
        # Mean=.5, dot products at lags 1 and 3 both equal .25.
        result = estimate_tempo([0, 0, 0, 1, 1, 0, 1, 1], 1, min_bpm=15, max_bpm=60)
        self.assertEqual(result.scores[0], result.scores[2])
        self.assertEqual(result.lag_frames, 1)
        self.assertEqual(result.bpm, 60)

    def test_invalid_inputs(self):
        for rate in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                estimate_tempo([0, 1], rate)
        for lower, upper in ((0, 200), (200, 100), (100, 100), (np.nan, 200), (60, np.inf)):
            with self.assertRaises(ValueError):
                estimate_tempo([0, 1], 64, min_bpm=lower, max_bpm=upper)
        for values in ([-1], [np.nan], [np.inf], [[0]], [1j]):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                estimate_tempo(values, 64)
