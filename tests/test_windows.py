"""Verify periodic Hann weights and gain-corrected spectra."""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from autodj.audio import save_spectrum
from autodj.audio.spectrum import _build_spectrum
from autodj.dsp import amplitude_spectrum, hann_window


class HannWindowTests(unittest.TestCase):
    def test_known_weights_and_singleton(self):
        np.testing.assert_allclose(hann_window(4), [0, 0.5, 1, 0.5], atol=1e-15)
        self.assertEqual(hann_window(1), (1,))
        self.assertEqual(hann_window(2), (0, 1))

    def test_matches_numpy_periodic_construction(self):
        for size in (3, 7, 16, 64):
            with self.subTest(size=size):
                weights = hann_window(size)
                np.testing.assert_allclose(weights, np.hanning(size + 1)[:-1], atol=1e-15)
                self.assertAlmostEqual(sum(weights) / size, 0.5)
                self.assertEqual(weights[0], 0)
                self.assertTrue(all(0 <= w <= 1 for w in weights))

    def test_invalid_size(self):
        for size in (0, -1, 2.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                hann_window(size)

    def test_preserves_centered_tone_amplitude_and_broadens_peak(self):
        samples = [2 * math.sin(2 * math.pi * 8 * n / 64) for n in range(64)]
        original = samples.copy()
        frequencies, amplitudes = amplitude_spectrum(samples, 1760, window="hann")
        self.assertEqual(frequencies[8], 220)
        self.assertAlmostEqual(amplitudes[8], 2)
        self.assertAlmostEqual(amplitudes[7], 1)
        self.assertAlmostEqual(amplitudes[9], 1)
        self.assertEqual(samples, original)

    def test_dc_nyquist_singleton_and_silence(self):
        self.assertAlmostEqual(amplitude_spectrum([3] * 16, 8000, window="hann")[1][0], 3)
        self.assertAlmostEqual(amplitude_spectrum([2, -2] * 8, 8000, window="hann")[1][-1], 2)
        self.assertEqual(amplitude_spectrum([-4], 8000, window="hann"), ((0,), (4,)))
        self.assertEqual(amplitude_spectrum([0] * 8, 8000, window="hann")[1], (0,) * 5)

    def test_matches_independent_fft_for_odd_and_even_lengths(self):
        for size in (2, 7, 31, 64):
            with self.subTest(size=size):
                samples = np.random.default_rng(42).normal(size=size)
                weights = np.hanning(size + 1)[:-1]
                expected = np.abs(np.fft.rfft(samples * weights)) / weights.sum()
                expected[1:-1 if size % 2 == 0 else None] *= 2
                frequencies, actual = amplitude_spectrum(samples, 8000, window="hann")
                np.testing.assert_allclose(frequencies, np.fft.rfftfreq(size, 1 / 8000))
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_reduces_distant_leakage_for_off_bin_tone(self):
        samples = [math.sin(2 * math.pi * 230 * n / 1760) for n in range(64)]
        frequencies, rectangular = amplitude_spectrum(samples, 1760)
        _, hann = amplitude_spectrum(samples, 1760, window="hann")
        # Exclude the wider Hann main lobe: measure only >2 bins from 230 Hz.
        far = [k for k, f in enumerate(frequencies) if abs(f - 230) > 2 * 27.5]
        self.assertLess(max(hann[k] for k in far), max(rectangular[k] for k in far))
        self.assertLess(max(hann), 1)  # Gain correction does not remove off-bin bias.

    def test_invalid_window_and_samples(self):
        with self.assertRaises(ValueError):
            amplitude_spectrum([1, 2], 8000, window="hamming")
        for samples in ([], [float("nan"), 1], [float("inf"), 1]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                amplitude_spectrum(samples, 8000, window="hann")

    def test_plot_uses_window_and_exports_png(self):
        samples = [math.sin(2 * math.pi * 8 * n / 64) for n in range(64)]
        figure = _build_spectrum(samples, 1760, "Hann", window="hann")
        self.addCleanup(figure.clear)
        markers = figure.axes[0].containers[0].markerline
        self.assertAlmostEqual(markers.get_ydata()[8], 1)
        self.assertAlmostEqual(markers.get_ydata()[7], 0.5)
        self.assertIn("Hann", figure.axes[0].get_title())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hann.png"
            save_spectrum(samples, 1760, path, window="hann")
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
