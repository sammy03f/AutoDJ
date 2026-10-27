"""Known magnitude changes and synthetic events with known sample positions."""

from dataclasses import replace
import unittest

import numpy as np

from autodj.analysis import spectral_flux
from autodj.dsp import stft


class SpectralFluxTests(unittest.TestCase):
    def result_with_coefficients(self, rows):
        result = stft([0] * (4 * len(rows)), 8, frame_size=4, hop_size=4)
        return replace(result, coefficients=np.asarray(rows, dtype=np.complex128))

    def test_positive_changes_are_summed_before_aggregation(self):
        result = self.result_with_coefficients([[1, 4, 0], [3, 1, 2], [2, 1, 5]])
        # Changes in row 1: +2, -3, +2 => 4, not max(0, sum(changes)) = 1.
        np.testing.assert_array_equal(spectral_flux(result), [0, 4, 3])

    def test_phase_changes_do_not_count(self):
        result = self.result_with_coefficients([[1, 2j, -3], [1j, -2, 3j]])
        np.testing.assert_array_equal(spectral_flux(result), [0, 0])

    def test_decreases_and_identical_frames_are_zero(self):
        result = self.result_with_coefficients([[5, 3, 1], [5, 3, 1], [1, 2, 0]])
        np.testing.assert_array_equal(spectral_flux(result), [0, 0, 0])

    def test_empty_single_frame_and_silence(self):
        for samples, expected in (([], []), ([1] * 4, [0]), ([0] * 12, [0, 0, 0])):
            with self.subTest(samples=samples):
                result = stft(samples, 8, frame_size=4, hop_size=4)
                np.testing.assert_array_equal(spectral_flux(result), expected)

    def test_synthetic_impulses_have_known_responses(self):
        samples = np.zeros(1024)
        samples[256], samples[768] = 1, 0.5
        result = stft(samples, 1024, frame_size=64, hop_size=16)
        expected = np.zeros(61)
        # An impulse has the same magnitude in all 33 bins. Its Hann weight
        # rises by 0.5 on each of these two frames; the second impulse is half gain.
        expected[[13, 14]] = 33 * 0.5
        expected[[45, 46]] = 33 * 0.25
        np.testing.assert_allclose(spectral_flux(result), expected, atol=1e-12)
        np.testing.assert_array_equal(result.times[[13, 14]], [208 / 1024, 224 / 1024])

    def test_stationary_bin_centered_tone_is_near_zero(self):
        samples = np.sin(2 * np.pi * 64 * np.arange(512) / 1024)
        result = stft(samples, 1024, frame_size=64, hop_size=16)
        np.testing.assert_allclose(spectral_flux(result), 0, atol=1e-10)

    def test_gain_scales_strength_and_does_not_change_input(self):
        samples = np.zeros(256)
        samples[128] = 1
        result = stft(samples, 1024, frame_size=64, hop_size=16)
        original = result.coefficients.copy()
        base = spectral_flux(result)
        scaled = stft(samples * 3, 1024, frame_size=64, hop_size=16)
        np.testing.assert_allclose(spectral_flux(scaled), 3 * base, atol=1e-12)
        np.testing.assert_array_equal(result.coefficients, original)
        base[:] = -1
        np.testing.assert_array_equal(result.coefficients, original)

    def test_invalid_shape_and_nonfinite_coefficients(self):
        result = self.result_with_coefficients([[1, 2, 3]])
        for coefficients in (np.zeros((1, 2)), np.zeros(3),
                             np.array([[np.nan, 0, 0]]), np.array([[np.inf, 0, 0]])):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                spectral_flux(replace(result, coefficients=coefficients))
