"""Audio file loading and sample access."""

from .loader import WavAudio, load_wav
from .gain import apply_gain
from .segments import select_time_range

__all__ = ["WavAudio", "load_wav", "select_time_range", "apply_gain"]
