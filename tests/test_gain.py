"""Verify gain against known integer samples and PCM boundaries."""

import unittest

from autodj.audio import WavAudio, apply_gain


class ApplyGainTests(unittest.TestCase):
    def test_attenuation_preserves_metadata_and_original(self):
        audio = WavAudio(8000, 2, ((100, -200), (300, -400)))
        result = apply_gain(audio, 0.5)
        self.assertEqual(result.samples, ((50, -100), (150, -200)))
        self.assertEqual(result.sample_rate, 8000)
        self.assertEqual(result.channels, 2)
        self.assertEqual(result.frame_count, 2)
        self.assertEqual(result.duration, audio.duration)
        self.assertEqual(audio.samples, ((100, -200), (300, -400)))

    def test_unity_preserves_extremes(self):
        audio = WavAudio(8000, 1, ((-32768,), (0,), (32767,)))
        self.assertEqual(apply_gain(audio, 1), audio)

    def test_zero_mutes_all_channels(self):
        audio = WavAudio(8000, 2, ((-32768, 32767), (12, -20)))
        self.assertEqual(apply_gain(audio, 0).samples, ((0, 0), (0, 0)))

    def test_amplifies_and_clips_both_limits(self):
        audio = WavAudio(8000, 1, ((100,), (-100,), (20000,), (-20000,)))
        self.assertEqual(
            apply_gain(audio, 2).samples, ((200,), (-200,), (32767,), (-32768,))
        )

    def test_rounds_halfway_values_to_even(self):
        audio = WavAudio(8000, 1, ((1,), (3,), (-1,), (-3,)))
        self.assertEqual(apply_gain(audio, 0.5).samples, ((0,), (2,), (0,), (-2,)))

    def test_large_finite_gain_clips_without_overflow(self):
        audio = WavAudio(8000, 1, ((32767,), (-32768,), (0,)))
        self.assertEqual(
            apply_gain(audio, 1e308).samples, ((32767,), (-32768,), (0,))
        )

    def test_empty_audio(self):
        audio = WavAudio(8000, 1, ())
        self.assertEqual(apply_gain(audio, 0.5), audio)

    def test_rejects_invalid_gain(self):
        audio = WavAudio(8000, 1, ((1,),))
        for gain in (-1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(gain=gain):
                with self.assertRaises(ValueError):
                    apply_gain(audio, gain)
