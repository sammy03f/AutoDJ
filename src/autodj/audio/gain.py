"""Adjust amplitude while keeping samples in the 16-bit PCM range."""

import math

from .loader import WavAudio


def apply_gain(audio: WavAudio, gain: float) -> WavAudio:
    """Return new audio with each sample multiplied by a nonnegative gain.

    Zero silences the audio; one preserves sample values. Reject negative
    or nonfinite gains with ValueError. Round to the nearest integer using
    Python's ties-to-even rule, and clip to [-32768, 32767]. Clipping can
    distort audio. Sample rate, channel order, and frame count are unchanged.
    """
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("Gain must be finite and nonnegative")

    # Clamp before rounding so even a very large gain cannot overflow round().
    samples = tuple(
        tuple(round(max(-32768, min(32767, sample * gain))) for sample in frame)
        for frame in audio.samples
    )
    return WavAudio(audio.sample_rate, audio.channels, samples)
