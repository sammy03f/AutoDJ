"""Test peak rules independently of STFT, then validate known bursts."""

import unittest

import numpy as np

from autodj.analysis import pick_onsets, spectral_flux
from autodj.dsp import stft


class PeakPickingTests(unittest.TestCase):
    def pick(self, values, threshold=0, min_interval=0, times=None):
        if times is None:
            times = np.arange(len(values), dtype=float)
        return pick_onsets(values, times, threshold=threshold, min_interval=min_interval)

    def test_strict_threshold_filters_weak_and_equal_peaks(self):
        result = self.pick([0, 1, 0, 2, 0, 3, 0], threshold=2)
        np.testing.assert_array_equal(result.frame_indices, [5])
        np.testing.assert_array_equal(result.times, [5])

    def test_silence_short_curves_and_boundaries(self):
        for values in ([], [5], [0, 5], [0, 0, 0], [5, 3, 0], [0, 3, 5], [5, 5, 5]):
            with self.subTest(values=values):
                result = self.pick(values)
                self.assertEqual(result.frame_indices.size, 0)
                self.assertEqual(result.times.size, 0)

    def test_plateaus_use_earliest_frame_and_need_two_lower_sides(self):
        np.testing.assert_array_equal(self.pick([0, 4, 4, 4, 0]).frame_indices, [1])
        for values in ([4, 4, 0], [0, 4, 4], [0, 3, 3, 5, 0]):
            with self.subTest(values=values):
                expected = [3] if len(values) == 5 else []
                np.testing.assert_array_equal(self.pick(values).frame_indices, expected)

    def test_stronger_close_peak_wins_even_if_later(self):
        result = self.pick([0, 3, 0, 5, 0, 2, 0], min_interval=3)
        np.testing.assert_array_equal(result.frame_indices, [3])

    def test_equal_peaks_favor_earlier_and_survivors_are_sorted(self):
        np.testing.assert_array_equal(self.pick([0, 4, 0, 4, 0], min_interval=3).frame_indices, [1])
        np.testing.assert_array_equal(self.pick([0, 2, 0, 0, 7, 0]).frame_indices, [1, 4])

    def test_interval_uses_seconds_and_allows_exact_boundary(self):
        values = [0, 2, 0, 3, 0]
        times = [0, 0.1, 0.2, 0.3, 0.4]
        np.testing.assert_array_equal(self.pick(values, times=times, min_interval=0.2).frame_indices, [1, 3])
        np.testing.assert_array_equal(self.pick(values, times=times, min_interval=0.201).frame_indices, [3])

    def test_irregular_times_are_used_without_correction(self):
        result = self.pick([0, 3, 0, 4, 0], times=[0, 0.25, 0.3, 1.0, 1.1], min_interval=0.5)
        np.testing.assert_array_equal(result.times, [0.25, 1])

    def test_input_and_output_do_not_alias(self):
        values = np.array([0.0, 5, 0])
        times = np.array([0.0, 0.5, 1])
        result = self.pick(values, times=times)
        result.times[0] = 999
        np.testing.assert_array_equal(values, [0, 5, 0])
        np.testing.assert_array_equal(times, [0, 0.5, 1])

    def test_invalid_parameters_and_arrays(self):
        for name in ("threshold", "min_interval"):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.pick([0, 1, 0], **{name: value})
        for values, times in (([0, 1], [0]), ([[0, 1]], [0]), ([0], [[0]]),
                              ([0, -1, 0], [0, 1, 2]), ([np.nan], [0]),
                              ([np.inf], [0]), ([0], [-1]), ([0], [np.nan]),
                              ([0], [np.inf]), ([0, 0], [1, 1]), ([0, 0], [1, 0])):
            with self.subTest(values=values, times=times), self.assertRaises(ValueError):
                self.pick(values, times=times)
        with self.assertRaises(TypeError):
            self.pick([0, 1j, 0])

    def test_known_bursts_have_three_uncorrected_frame_start_estimates(self):
        samples = np.zeros(3072)
        starts = np.array([512, 1536, 2560])
        n = np.arange(128)
        burst = np.cos(2 * np.pi * 128 * n / 2048) * np.exp(-n / 24)
        for start, gain in zip(starts, [1, 0.6, 0.8]):
            samples[start:start + 128] = gain * burst
        analysis = stft(samples, 2048, frame_size=128, hop_size=32)
        result = self.pick(spectral_flux(analysis), times=analysis.times,
                           threshold=10, min_interval=0.1)
        # The maximal increase occurs in the frame beginning two hops before
        # each burst; picking a peak does not shift it to the window center.
        np.testing.assert_array_equal(result.frame_indices, [14, 46, 78])
        np.testing.assert_array_equal(result.times - starts / 2048, [-1 / 32] * 3)
