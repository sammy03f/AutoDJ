"""Write integer samples as uncompressed 16-bit PCM WAV audio."""

from pathlib import Path
import struct
import wave

from .loader import WavAudio


def save_wav(audio: WavAudio, path: str | Path) -> None:
    """Save audio, preserving sample values, sample rate, and channel order.

    The parent directory must exist. Existing files are overwritten. Invalid
    metadata, frame shapes, or samples raise ValueError before opening the
    destination; filesystem errors propagate. Samples must be integers in
    [-32768, 32767]. This version builds the PCM bytes in memory.
    """
    if type(audio.channels) is not int or not 1 <= audio.channels <= 32767:
        raise ValueError("Channels must be an integer from 1 to 32767")
    if (
        type(audio.sample_rate) is not int
        or audio.sample_rate <= 0
        or audio.sample_rate * audio.channels * 2 > 0xFFFFFFFF
    ):
        raise ValueError("Sample rate must be a positive integer fitting the WAV header")
    if audio.frame_count * audio.channels * 2 > 0xFFFFFFFF - 36:
        raise ValueError("Audio is too large for a standard RIFF WAV file")

    for frame in audio.samples:
        if len(frame) != audio.channels:
            raise ValueError("Each frame must contain one sample per channel")
        for sample in frame:
            if type(sample) is not int or not -32768 <= sample <= 32767:
                raise ValueError("Samples must be integers in the 16-bit PCM range")

    pack_frame = struct.Struct("<" + "h" * audio.channels).pack
    raw = b"".join(pack_frame(*frame) for frame in audio.samples)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(audio.channels)
        output.setsampwidth(2)
        output.setframerate(audio.sample_rate)
        output.writeframes(raw)
