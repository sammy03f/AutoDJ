"""Check time selection against small signals with known frame positions."""

import unittest

from autodj.audio import WavAudio, select_time_range


class SelectTimeRangeTests(unittest.TestCase):
    def setUp(self):
        self.audio = WavAudio(4, 1, ((0,), (10,), (20,), (30,), (40,), (50,)))

    def test_selects_start_inclusive_end_exclusive(self):
        clip = select_time_range(self.audio, 0.25, 1.0)
        self.assertEqual(clip.samples, ((10,), (20,), (30,)))
        self.assertEqual(clip.sample_rate, 4)
        self.assertEqual(clip.channels, 1)
        self.assertEqual(clip.frame_count, 3)
        self.assertEqual(clip.duration, 0.75)
        self.assertEqual(self.audio.frame_count, 6)

    def test_preserves_stereo_channels(self):
        audio = WavAudio(2, 2, ((1, -1), (2, -2), (3, -3)))
        clip = select_time_range(audio, 0.5, 1.5)
        self.assertEqual(clip.samples, ((2, -2), (3, -3)))
        self.assertEqual(clip.channels, 2)

    def test_rounds_fractional_frames_down(self):
        clip = select_time_range(self.audio, 0.3, 0.9)
        self.assertEqual(clip.samples, ((10,), (20,)))
        self.assertEqual(clip.duration, 0.5)

    def test_full_duration_keeps_final_frame(self):
        # This duration times the rate is slightly less than 15 in float math.
        audio = WavAudio(44100, 1, tuple((n,) for n in range(15)))
        self.assertEqual(select_time_range(audio, 0, audio.duration), audio)

    def test_equal_boundaries_are_empty(self):
        for time in (0, 0.3, self.audio.duration):
            with self.subTest(time=time):
                clip = select_time_range(self.audio, time, time)
                self.assertEqual(clip.samples, ())
                self.assertEqual(clip.duration, 0)

    def test_empty_input(self):
        audio = WavAudio(8000, 1, ())
        self.assertEqual(select_time_range(audio, 0, 0), audio)

    def test_rejects_invalid_ranges(self):
        for start, end in (
            (-0.1, 1), (1, 0.5), (0, 2), (2, 2),
            (float("nan"), 1), (0, float("nan")),
            (float("-inf"), 1), (0, float("inf")),
        ):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    select_time_range(self.audio, start, end)
