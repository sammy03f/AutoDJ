"""Select audio frames using times in seconds."""

import math

from .loader import WavAudio


def select_time_range(audio: WavAudio, start: float, end: float) -> WavAudio:
    """Return a new clip spanning start (inclusive) to end (exclusive).

    Require finite times with 0 <= start <= end <= audio.duration.
    Convert seconds to frame indices by rounding down time * sample_rate.
    The exact duration maps to frame_count to avoid floating-point rounding
    dropping the final frame. Equal boundaries produce an empty clip.
    Samples and channel order are preserved; the input is unchanged.
    """
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Start and end must be finite times in seconds")
    if not 0 <= start <= end <= audio.duration:
        raise ValueError("Require 0 <= start <= end <= audio duration")

    start_frame = (
        audio.frame_count if start == audio.duration
        else math.floor(start * audio.sample_rate)
    )
    end_frame = (
        audio.frame_count if end == audio.duration
        else math.floor(end * audio.sample_rate)
    )
    return WavAudio(
        audio.sample_rate, audio.channels, audio.samples[start_frame:end_frame]
    )
