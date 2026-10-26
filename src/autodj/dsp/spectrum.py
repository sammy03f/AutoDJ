"""Convert a short real signal into a one-sided amplitude spectrum."""

from collections.abc import Iterable
import math
from numbers import Complex, Real
from typing import Literal

from .fourier import dft
from .windows import hann_window


def amplitude_spectrum(
    samples: Iterable[float], sample_rate: float, *, window: Literal["hann"] | None = None
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (frequencies_hz, amplitudes) for a nonempty real signal.

    Use the educational O(N**2) DFT, without zero padding. Default window=None
    preserves rectangular-window behavior. window="hann" multiplies samples
    by a periodic Hann, then divides by sum(weights) instead of N to correct
    coherent gain. Keep bins 0 through N//2, doubling every bin except DC and the
    Nyquist bin (which exists only for even N). Amplitude uses the input's
    units: integer PCM samples produce PCM amplitude, not normalized audio.

    Tone amplitudes are accurate for isolated, bin-centered sinusoids away from
    overlapping positive/negative main lobes. Off-bin peaks can underestimate
    amplitude even after window correction. Empty/nonfinite samples and nonpositive/nonfinite sample
    rates raise ValueError. Complex samples are unsupported. Use short inputs.
    """
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be finite and positive")
    if window not in (None, "hann"):
        raise ValueError("Window must be None or 'hann'")
    values = tuple(samples)
    if any(isinstance(sample, Complex) and not isinstance(sample, Real) for sample in values):
        raise TypeError("Amplitude spectrum requires real samples")
    values = tuple(float(sample) for sample in values)
    size = len(values)
    normalization = size
    if window == "hann":
        weights = hann_window(size)
        values = tuple(sample * weight for sample, weight in zip(values, weights))
        normalization = sum(weights)
    coefficients = dft(values)
    frequencies = tuple(k * (sample_rate / size) for k in range(size // 2 + 1))
    amplitudes = []
    for k in range(len(frequencies)):
        amplitude = abs(coefficients[k]) / normalization
        if k != 0 and not (size % 2 == 0 and k == size // 2):
            amplitude *= 2
        amplitudes.append(amplitude)
    return frequencies, tuple(amplitudes)
