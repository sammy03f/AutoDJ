"""Check framing, phase, timestamps, and a known change in tone."""

import unittest

import numpy as np

from autodj.dsp import dft, stft


class StftTests(unittest.TestCase):
    def test_known_frames_check_phase_and_unnormalized_scale(self):
        result = stft(range(7), 8, frame_size=4, hop_size=2)
        np.testing.assert_allclose(result.coefficients, [[4, -2 + 1j, 0], [8, -4 + 1j, 0]], atol=1e-14)
        np.testing.assert_array_equal(result.times, [0, 0.25])
        np.testing.assert_array_equal(result.frequencies, [0, 2, 4])
        self.assertEqual(result.window_sum, 2)
        self.assertEqual((result.sample_rate, result.frame_size, result.hop_size), (8, 4, 2))

    def test_matches_direct_dft_for_odd_and_even_frames(self):
        samples = np.random.default_rng(42).normal(size=23)
        for size, hop in ((5, 2), (8, 3)):
            with self.subTest(size=size):
                result = stft(samples, 8000, frame_size=size, hop_size=hop)
                weights = np.hanning(size + 1)[:-1]
                expected = [dft(samples[start:start + size] * weights)[:size // 2 + 1]
                            for start in range(0, len(samples) - size + 1, hop)]
                np.testing.assert_allclose(result.coefficients, expected, atol=1e-12)
                np.testing.assert_allclose(result.frequencies, np.fft.rfftfreq(size, 1 / 8000))

    def test_tracks_tone_switch_in_complete_unmixed_frames(self):
        times = np.arange(256) / 1024
        samples = np.concatenate((np.sin(2 * np.pi * 64 * times),
                                  0.5 * np.sin(2 * np.pi * 192 * times)))
        result = stft(samples, 1024, frame_size=64, hop_size=32)
        self.assertEqual(result.coefficients.shape, (15, 33))
        np.testing.assert_array_equal(result.times, np.arange(15) / 32)
        peaks = np.argmax(np.abs(result.coefficients), axis=1)
        # Frame 7 crosses the switch at sample 256; neither pure-tone assertion applies.
        np.testing.assert_array_equal(result.frequencies[peaks[:7]], [64] * 7)
        np.testing.assert_array_equal(result.frequencies[peaks[8:]], [192] * 7)
        np.testing.assert_allclose(2 * np.abs(result.coefficients[:7, 4]) / result.window_sum, 1)
        np.testing.assert_allclose(2 * np.abs(result.coefficients[8:, 12]) / result.window_sum, 0.5)

    def test_empty_and_short_inputs(self):
        for samples in ([], [1, 2, 3]):
            with self.subTest(samples=samples):
                result = stft(samples, 8000, frame_size=4, hop_size=2)
                self.assertEqual(result.coefficients.shape, (0, 3))
                self.assertEqual(result.times.size, 0)
                np.testing.assert_array_equal(result.frequencies, [0, 2000, 4000])

    def test_exact_fit_no_overlap_and_incomplete_tail(self):
        for length, expected in ((4, 1), (7, 1), (8, 2), (9, 2)):
            with self.subTest(length=length):
                result = stft([1] * length, 8, frame_size=4, hop_size=4)
                self.assertEqual(result.coefficients.shape, (expected, 3))
        a = stft([1] * 8, 8, frame_size=4, hop_size=4)
        b = stft([1] * 8 + [999], 8, frame_size=4, hop_size=4)
        np.testing.assert_array_equal(a.coefficients, b.coefficients)

    def test_single_sample_frames(self):
        result = stft([2, -3], 8, frame_size=1, hop_size=1)
        np.testing.assert_array_equal(result.coefficients, [[2], [-3]])
        np.testing.assert_array_equal(result.times, [0, 0.125])
        self.assertEqual(result.window_sum, 1)

    def test_silence(self):
        result = stft([0] * 16, 8000, frame_size=8, hop_size=4)
        np.testing.assert_array_equal(result.coefficients, np.zeros((3, 5)))

    def test_array_and_iterator_inputs_are_preserved(self):
        samples = np.arange(8, dtype=np.int16)
        original = samples.copy()
        array_result = stft(samples, 8, frame_size=4, hop_size=2)
        iterator_result = stft(iter(samples), 8, frame_size=4, hop_size=2)
        np.testing.assert_array_equal(array_result.coefficients, iterator_result.coefficients)
        np.testing.assert_array_equal(samples, original)

    def test_invalid_sizes(self):
        for frame, hop in ((0, 1), (-1, 1), (2.5, 1), (True, 1),
                           (4, 0), (4, -1), (4, 5), (4, 1.5), (4, True)):
            with self.subTest(frame=frame, hop=hop), self.assertRaises(ValueError):
                stft([], 8000, frame_size=frame, hop_size=hop)

    def test_invalid_rates_and_samples(self):
        for rate in (0, -1, float("inf"), float("nan")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                stft([1], rate, frame_size=1, hop_size=1)
        for samples in ([float("nan")], [float("inf")], [[1, 2]], [0] * 4 + [float("nan")]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                stft(samples, 8000, frame_size=4, hop_size=4)
        with self.assertRaises(TypeError):
            stft([1 + 1j], 8000, frame_size=1, hop_size=1)
