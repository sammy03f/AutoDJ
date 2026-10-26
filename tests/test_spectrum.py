"""Check frequency mapping, amplitude scaling, and actual PNG export."""

import math
from pathlib import Path
import tempfile
import unittest

from matplotlib import image
import numpy as np

from autodj.audio import save_spectrum
from autodj.audio.spectrum import _build_spectrum
from autodj.dsp import amplitude_spectrum


class AmplitudeSpectrumTests(unittest.TestCase):
    def test_two_tone_frequencies_and_amplitudes(self):
        samples = [math.sin(2 * math.pi * 8 * n / 64)
                   + 0.5 * math.sin(2 * math.pi * 16 * n / 64) for n in range(64)]
        frequencies, amplitudes = amplitude_spectrum(samples, 1760)
        self.assertEqual(frequencies, tuple(k * 27.5 for k in range(33)))
        expected = [0.0] * 33
        expected[8], expected[16] = 1, 0.5
        np.testing.assert_allclose(amplitudes, expected, atol=1e-12)

    def test_dc_and_nyquist_are_not_doubled(self):
        np.testing.assert_allclose(amplitude_spectrum([-3] * 8, 8000)[1],
                                   [3, 0, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(amplitude_spectrum([2, -2] * 4, 8000)[1],
                                   [0, 0, 0, 0, 2], atol=1e-12)

    def test_highest_odd_length_bin_is_doubled(self):
        samples = [3 * math.cos(2 * math.pi * 3 * n / 7) for n in range(7)]
        frequencies, amplitudes = amplitude_spectrum(samples, 700)
        self.assertEqual(frequencies, (0, 100, 200, 300))
        np.testing.assert_allclose(amplitudes, [0, 0, 0, 3], atol=1e-12)

    def test_single_sample_and_two_samples(self):
        self.assertEqual(amplitude_spectrum([-4], 8000), ((0,), (4,)))
        frequencies, amplitudes = amplitude_spectrum([1, -1], 8000)
        self.assertEqual(frequencies, (0, 4000))
        np.testing.assert_allclose(amplitudes, [0, 1], atol=1e-12)

    def test_silence(self):
        self.assertEqual(amplitude_spectrum([0] * 8, 8000)[1], (0,) * 5)

    def test_pcm_units_and_input_preserved(self):
        samples = [0, 12000, 0, -12000]
        frequencies, amplitudes = amplitude_spectrum(iter(samples), 8000)
        self.assertEqual(frequencies, (0, 2000, 4000))
        np.testing.assert_allclose(amplitudes, [0, 12000, 0], atol=1e-10)
        self.assertEqual(samples, [0, 12000, 0, -12000])

    def test_invalid_inputs(self):
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                amplitude_spectrum([1], rate)
        for samples in ([], [float("nan")], [float("inf")]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                amplitude_spectrum(samples, 8000)
        for sample in (1 + 1j, np.complex64(1 + 1j)):
            with self.subTest(sample=sample), self.assertRaises(TypeError):
                amplitude_spectrum([sample], 8000)


class SpectrumPlotTests(unittest.TestCase):
    def test_plotted_data_and_axis_units(self):
        figure = _build_spectrum([0, 2, 0, -2], 800, "Test")
        self.addCleanup(figure.clear)
        axis = figure.axes[0]
        markers = axis.containers[0].markerline
        np.testing.assert_allclose(markers.get_xdata(), [0, 200, 400])
        np.testing.assert_allclose(markers.get_ydata(), [0, 2, 0], atol=1e-12)
        self.assertEqual(axis.get_xlabel(), "Frequency (Hz)")
        self.assertEqual(axis.get_ylabel(), "Amplitude (input units)")
        self.assertEqual(axis.get_xlim(), (0, 400))

    def test_png_export_for_tone_silence_and_single_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrum.png"
            for samples in ([0, 1, 0, -1], [0] * 8, [2]):
                with self.subTest(samples=samples):
                    save_spectrum(samples, 8000, str(path))
                    self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    pixels = image.imread(path)
                    self.assertEqual(pixels.shape[1], 1200)
                    self.assertGreater(float(pixels.max() - pixels.min()), 0.5)

    def test_invalid_samples_preserve_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrum.png"
            path.write_bytes(b"original")
            with self.assertRaises(ValueError):
                save_spectrum([], 8000, path)
            self.assertEqual(path.read_bytes(), b"original")

    def test_invalid_path(self):
        with self.assertRaises(ValueError):
            save_spectrum([1], 8000, "spectrum.wav")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                save_spectrum([1], 8000, Path(directory) / "missing" / "spectrum.png")
