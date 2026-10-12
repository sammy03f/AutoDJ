"""Check both round trips and independently decoded WAV bytes."""

from pathlib import Path
import tempfile
import unittest
import wave

from autodj.audio import WavAudio, apply_gain, load_wav, save_wav, select_time_range


class SaveWavTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "saved.wav"

    def test_pcm_bytes_and_header(self):
        audio = WavAudio(44100, 2, ((-32768, 32767), (-1, 0)))
        save_wav(audio, self.path)
        with wave.open(str(self.path), "rb") as saved:
            self.assertEqual(saved.getnchannels(), 2)
            self.assertEqual(saved.getframerate(), 44100)
            self.assertEqual(saved.getsampwidth(), 2)
            self.assertEqual(saved.getnframes(), 2)
            self.assertEqual(saved.getcomptype(), "NONE")
            self.assertEqual(saved.readframes(2), b"\x00\x80\xff\x7f\xff\xff\x00\x00")

    def test_round_trip_mono_stereo_and_empty(self):
        for rate, channels, samples in (
            (8000, 1, ((-32768,), (0,), (32767,))),
            (48000, 2, ((100, -100), (200, -200))),
            (44100, 2, ()),
        ):
            with self.subTest(rate=rate, channels=channels):
                audio = WavAudio(rate, channels, samples)
                save_wav(audio, str(self.path))
                self.assertEqual(load_wav(self.path), audio)

    def test_load_select_gain_save_workflow(self):
        save_wav(WavAudio(4, 1, ((100,), (20000,), (-20000,), (0,))), self.path)
        clip = select_time_range(load_wav(self.path), 0.25, 0.75)
        save_wav(apply_gain(clip, 2), self.path)
        self.assertEqual(load_wav(self.path), WavAudio(4, 1, ((32767,), (-32768,))))

    def test_overwriting_shorter_audio_removes_old_frames(self):
        save_wav(WavAudio(8000, 1, ((1,),) * 100), self.path)
        audio = WavAudio(8000, 1, ((2,),))
        save_wav(audio, self.path)
        self.assertEqual(load_wav(self.path), audio)
        self.assertEqual(self.path.stat().st_size, 46)

    def test_invalid_audio_does_not_overwrite_destination(self):
        invalid = (
            WavAudio(0, 1, ()), WavAudio(1.5, 1, ()),
            WavAudio(2**32, 1, ()), WavAudio(8000, 0, ()),
            WavAudio(8000, 32768, ()), WavAudio(8000, 1.5, ()),
            WavAudio(8000, 2, ((1,),)), WavAudio(8000, 1, ((1, 2),)),
            WavAudio(8000, 1, ((32768,),)), WavAudio(8000, 1, ((-32769,),)),
            WavAudio(8000, 1, ((1.5,),)), WavAudio(8000, 1, ((True,),)),
        )
        self.path.write_bytes(b"keep this file")
        for audio in invalid:
            with self.subTest(audio=audio):
                with self.assertRaises(ValueError):
                    save_wav(audio, self.path)
                self.assertEqual(self.path.read_bytes(), b"keep this file")

    def test_missing_parent_directory(self):
        with self.assertRaises(FileNotFoundError):
            save_wav(WavAudio(8000, 1, ()), self.path / "missing.wav")
