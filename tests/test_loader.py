"""Use generated PCM values so expected samples are known exactly."""

from pathlib import Path
import struct
import tempfile
import unittest
import wave

from autodj.audio import load_wav


class LoadWavTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "signal.wav"

    def write_wav(self, channels, values, sample_width=2):
        with wave.open(str(self.path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(sample_width)
            output.setframerate(8000)
            output.writeframes(struct.pack("<" + "h" * len(values), *values))

    def test_mono_samples_and_metadata(self):
        self.write_wav(1, [-32768, -1, 0, 32767])
        audio = load_wav(self.path)
        self.assertEqual(audio.sample_rate, 8000)
        self.assertEqual(audio.channels, 1)
        self.assertEqual(audio.samples, ((-32768,), (-1,), (0,), (32767,)))
        self.assertEqual(audio.frame_count, 4)
        self.assertEqual(audio.duration, 4 / 8000)

    def test_stereo_channel_order_and_duration(self):
        self.write_wav(2, [100, -100, 200, -200])
        audio = load_wav(str(self.path))
        self.assertEqual(audio.channels, 2)
        self.assertEqual(audio.samples, ((100, -100), (200, -200)))
        self.assertEqual(audio.frame_count, 2)
        self.assertEqual(audio.duration, 2 / 8000)

    def test_empty_audio(self):
        self.write_wav(1, [])
        audio = load_wav(self.path)
        self.assertEqual(audio.samples, ())
        self.assertEqual(audio.duration, 0)

    def test_rejects_other_bit_depths(self):
        self.write_wav(1, [0], sample_width=1)
        with self.assertRaisesRegex(ValueError, "16-bit PCM"):
            load_wav(self.path)

    def test_rejects_truncated_samples(self):
        self.write_wav(1, [100, 200])
        self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "truncated"):
            load_wav(self.path)

    def test_rejects_invalid_file(self):
        self.path.write_bytes(b"not a WAV file")
        with self.assertRaises(ValueError):
            load_wav(self.path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_wav(self.path)
