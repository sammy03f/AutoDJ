"""Known-answer tests plus comparison against an independent FFT."""

import cmath
import math
import unittest

import numpy as np

from autodj.dsp import dft


class DftTests(unittest.TestCase):
    def assertSpectrum(self, samples, expected):
        np.testing.assert_allclose(dft(samples), expected, rtol=1e-10, atol=1e-10)

    def test_silence_and_single_sample(self):
        self.assertSpectrum([0] * 8, [0j] * 8)
        self.assertEqual(dft([3 + 2j]), (3 + 2j,))

    def test_constant_signal_is_dc(self):
        self.assertSpectrum([2] * 4, [8, 0, 0, 0])

    def test_impulse_has_equal_coefficients(self):
        self.assertSpectrum([1, 0, 0, 0], [1, 1, 1, 1])

    def test_shifted_impulse_checks_phase_and_sign(self):
        self.assertSpectrum([0, 1, 0, 0], [1, -1j, -1, 1j])

    def test_sine_cosine_and_nyquist(self):
        self.assertSpectrum([0, 1, 0, -1], [0, -2j, 0, 2j])
        self.assertSpectrum([1, 0, -1, 0], [0, 2, 0, 2])
        self.assertSpectrum([1, -1, 1, -1], [0, 0, 4, 0])

    def test_two_tones_have_expected_amplitudes(self):
        # Fs=1760 Hz, N=64: 27.5 Hz/bin, so 220 Hz is bin 8, 440 Hz bin 16.
        samples = [
            math.sin(2 * math.pi * 8 * n / 64)
            + 0.5 * math.sin(2 * math.pi * 16 * n / 64)
            for n in range(64)
        ]
        expected = [0j] * 64
        expected[8], expected[56] = -32j, 32j
        expected[16], expected[48] = -16j, 16j
        self.assertSpectrum(samples, expected)

    def test_complex_tone_has_one_peak(self):
        samples = [cmath.exp(2j * math.pi * 2 * n / 7) for n in range(7)]
        self.assertSpectrum(samples, [0, 0, 7, 0, 0, 0, 0])

    def test_matches_fft_for_varied_lengths_and_inputs(self):
        random = np.random.default_rng(42)
        for length in (1, 2, 3, 7, 16, 31, 64):
            for complex_input in (False, True):
                with self.subTest(length=length, complex_input=complex_input):
                    samples = random.normal(size=length)
                    if complex_input:
                        samples = samples + 1j * random.normal(size=length)
                    self.assertSpectrum(samples, np.fft.fft(samples))

    def test_iterable_input_is_not_modified(self):
        samples = [1, 0, 0, 0]
        self.assertSpectrum(iter(samples), [1, 1, 1, 1])
        self.assertEqual(samples, [1, 0, 0, 0])

    def test_rejects_empty_and_nonfinite_inputs(self):
        for samples in ([], [float("nan")], [float("inf")],
                        [float("-inf")], [complex(0, float("nan"))]):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    dft(samples)
