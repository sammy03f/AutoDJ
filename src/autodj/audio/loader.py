"""Load uncompressed 16-bit PCM WAV files without resampling."""

from dataclasses import dataclass
from pathlib import Path
import struct
import wave


@dataclass(frozen=True)
class WavAudio:
    """Integer PCM samples arranged as frames, each containing all channels.

    Sample values range from -32768 to 32767. For stereo audio,
    samples[frame] is (left, right); mono frames contain one value.
    """

    sample_rate: int
    channels: int
    samples: tuple[tuple[int, ...], ...]

    @property
    def frame_count(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.frame_count / self.sample_rate


def load_wav(path: str | Path) -> WavAudio:
    """Read 16-bit PCM audio, preserving sample rate and channel order.

    Raise ValueError for unsupported or malformed WAV data. Filesystem
    errors, including FileNotFoundError, propagate to the caller.
    The entire file is loaded into memory; this first version targets
    small files rather than streaming full DJ sets.
    """
    try:
        with wave.open(str(path), "rb") as source:
            if source.getcomptype() != "NONE" or source.getsampwidth() != 2:
                raise ValueError("Only uncompressed 16-bit PCM WAV files are supported")
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            if sample_rate <= 0:
                raise ValueError("WAV sample rate must be positive")
            raw = source.readframes(frame_count)
            if len(raw) != frame_count * channels * 2:
                raise ValueError("WAV sample data is truncated")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Invalid or unsupported WAV file") from exc

    # WAV PCM is little-endian; 'h' decodes one signed 16-bit integer.
    frames = tuple(struct.iter_unpack("<" + "h" * channels, raw))
    return WavAudio(sample_rate, channels, frames)
