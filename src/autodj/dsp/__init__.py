"""Small, explainable digital signal processing building blocks."""

from .fourier import dft
from .spectrum import amplitude_spectrum

__all__ = ["dft", "amplitude_spectrum"]
