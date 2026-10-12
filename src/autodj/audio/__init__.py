"""Audio file loading and sample access."""

from .loader import WavAudio, load_wav
from .gain import apply_gain
from .segments import select_time_range
from .writer import save_wav

__all__ = ["WavAudio", "load_wav", "save_wav", "select_time_range", "apply_gain"]
