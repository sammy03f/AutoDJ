"""Check amplitude scaling, plot coordinates, orientation, and PNG rendering."""

from pathlib import Path
import tempfile
import unittest

from matplotlib import image
import numpy as np

from autodj.audio import save_spectrogram
from autodj.audio.spectrogram import _build_spectrogram, _spectrogram_db
from autodj.dsp import stft


class SpectrogramTests(unittest.TestCase):
    def test_tone_switch_levels_and_input_preservation(self):
        t = np.arange(256) / 1024
        samples = np.concatenate((np.sin(2 * np.pi * 64 * t),
                                  0.5 * np.sin(2 * np.pi * 192 * t)))
        result = stft(samples, 1024, frame_size=64, hop_size=32)
        original = result.coefficients.copy()
        db = _spectrogram_db(result, 1, -80)
        np.testing.assert_allclose(db[:7, 4], 0, atol=1e-12)
        np.testing.assert_allclose(db[8:, 12], 20 * np.log10(0.5), atol=1e-12)
        np.testing.assert_array_equal(result.coefficients, original)
        np.testing.assert_array_equal(db[:7, 12], [-80] * 7)

    def test_dc_and_nyquist_not_doubled(self):
        for samples, bin_index in (([0.5] * 8, 0), ([0.5, -0.5] * 4, 4)):
            with self.subTest(bin_index=bin_index):
                result = stft(samples, 8, frame_size=8, hop_size=8)
                self.assertAlmostEqual(_spectrogram_db(result, 1, -80)[0, bin_index],
                                       20 * np.log10(0.5))

    def test_highest_odd_bin_is_doubled(self):
        # At size 3 the sole interior coefficient has a known raw magnitude of 0.75.
        result = stft([0, 1, 0], 3, frame_size=3, hop_size=3)
        self.assertAlmostEqual(_spectrogram_db(result, 1, -80)[0, 1], 0)

    def test_pcm_reference_and_color_saturation(self):
        result = stft([16384] * 8, 8, frame_size=8, hop_size=8)
        self.assertAlmostEqual(_spectrogram_db(result, 32768, -80)[0, 0], 20 * np.log10(0.5))
        self.assertEqual(_spectrogram_db(result, 1, -80)[0, 0], 0)
        quiet = stft([1e-6], 8, frame_size=1, hop_size=1)
        self.assertEqual(_spectrogram_db(quiet, 1, -60)[0, 0], -60)

    def test_silence_is_finite_display_floor(self):
        result = stft([0] * 16, 8, frame_size=8, hop_size=4)
        with np.errstate(all="raise"):
            db = _spectrogram_db(result, 1, -80)
        np.testing.assert_array_equal(db, np.full((3, 5), -80))

    def test_mesh_orientation_and_time_frequency_edges(self):
        result = stft(range(8), 8, frame_size=4, hop_size=2)
        figure = _build_spectrogram(result, "Test", reference_amplitude=16)
        self.addCleanup(figure.clear)
        axis = figure.axes[0]
        mesh = axis.collections[0]
        np.testing.assert_allclose(np.asarray(mesh.get_array()).reshape(3, 3),
                                   _spectrogram_db(result, 16, -80).T)
        coordinates = mesh.get_coordinates()
        np.testing.assert_allclose(coordinates[0, :, 0], [0, 0.25, 0.5, 0.75])
        np.testing.assert_allclose(coordinates[:, 0, 1], [0, 1, 3, 4])
        self.assertEqual(axis.get_xlabel(), "Frame start time (seconds)")
        self.assertEqual(mesh.get_clim(), (-80, 0))

    def test_png_for_single_frame_odd_frame_and_silence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrogram.png"
            for samples, size in (([2], 1), ([0] * 16, 8), ([0, 1, 0, -1, 0], 5)):
                with self.subTest(size=size):
                    result = stft(samples, 8, frame_size=size, hop_size=size)
                    save_spectrogram(result, str(path))
                    self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    pixels = image.imread(path)
                    self.assertEqual(pixels.shape[1], 1200)

    def test_invalid_display_settings(self):
        result = stft([1], 8, frame_size=1, hop_size=1)
        for reference in (0, -1, float("inf"), float("nan")):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                _spectrogram_db(result, reference, -80)
        for floor in (0, 1, float("inf"), float("nan")):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                _spectrogram_db(result, 1, floor)

    def test_empty_result_preserves_existing_file(self):
        result = stft([], 8, frame_size=4, hop_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.png"
            path.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "complete STFT frame"):
                save_spectrogram(result, path)
            self.assertEqual(path.read_bytes(), b"keep")

    def test_invalid_paths(self):
        result = stft([1], 8, frame_size=1, hop_size=1)
        with self.assertRaises(ValueError):
            save_spectrogram(result, "invalid.wav")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                save_spectrogram(result, Path(directory) / "missing" / "plot.png")
