"""Verify plotted data and scales, plus real PNG rendering."""

from pathlib import Path
import tempfile
import unittest

from matplotlib import image

from autodj.audio import WavAudio, apply_gain, save_waveform
from autodj.audio.waveform import _build_waveform


class WaveformTests(unittest.TestCase):
    def build(self, audio):
        figure = _build_waveform(audio, "Test waveform")
        self.addCleanup(figure.clear)
        return figure

    def test_timestamps_and_channel_order(self):
        audio = WavAudio(4, 2, ((100, -100), (200, -200), (300, -300)))
        figure = self.build(audio)
        self.assertEqual(len(figure.axes), 2)
        for axis, expected in zip(figure.axes, ([100, 200, 300], [-100, -200, -300])):
            self.assertEqual(list(axis.lines[0].get_xdata()), [0, 0.25, 0.5])
            self.assertEqual(list(axis.lines[0].get_ydata()), expected)
            self.assertEqual(axis.get_xlim(), (0, 0.75))
        self.assertEqual(figure.axes[0].get_title(loc="left"), "Left")
        self.assertEqual(figure.axes[1].get_title(loc="left"), "Right")
        self.assertEqual(figure.axes[1].get_xlabel(), "Time (seconds)")

    def test_gain_variants_keep_same_scale_and_visible_clipping(self):
        original = WavAudio(8000, 1, ((20000,), (-20000,)))
        figures = [self.build(apply_gain(original, gain)) for gain in (1, 0.5, 2)]
        for figure in figures:
            self.assertEqual(figure.axes[0].get_ylim(), figures[0].axes[0].get_ylim())
        self.assertEqual(list(figures[1].axes[0].lines[0].get_ydata()), [10000, -10000])
        self.assertEqual(list(figures[2].axes[0].lines[0].get_ydata()), [32767, -32768])
        self.assertEqual(original.samples, ((20000,), (-20000,)))

    def test_single_frame_is_visible_at_zero(self):
        axis = self.build(WavAudio(8000, 1, ((100,),))).axes[0]
        self.assertEqual(list(axis.lines[0].get_xdata()), [0])
        self.assertEqual(axis.lines[0].get_marker(), ".")
        self.assertEqual(axis.get_xlim(), (0, 1 / 8000))

    def test_empty_audio_is_annotated(self):
        axis = self.build(WavAudio(8000, 1, ())).axes[0]
        self.assertEqual(len(axis.lines[0].get_xdata()), 0)
        self.assertIn("No samples", axis.texts[0].get_text())

    def test_generic_channel_labels(self):
        figure = self.build(WavAudio(8000, 3, ((1, 2, 3),)))
        self.assertEqual(figure.axes[2].get_title(loc="left"), "Channel 3")

    def test_real_png_output_for_mono_stereo_silence_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waveform.png"
            for audio in (
                WavAudio(4, 1, ((-32768,), (0,), (32767,))),
                WavAudio(4, 2, ((100, -100), (200, -200))),
                WavAudio(4, 1, ((0,),) * 4), WavAudio(4, 1, ()),
            ):
                with self.subTest(channels=audio.channels, frames=audio.frame_count):
                    save_waveform(audio, str(path))
                    self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                    pixels = image.imread(path)
                    self.assertEqual(pixels.shape[1], 1200)
                    self.assertGreater(float(pixels.max() - pixels.min()), 0.5)

    def test_rejects_non_png_path(self):
        with self.assertRaisesRegex(ValueError, ".png"):
            save_waveform(WavAudio(8000, 1, ()), "waveform.wav")

    def test_missing_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                save_waveform(WavAudio(8000, 1, ()), Path(directory) / "missing" / "plot.png")
