"""Convert a short real signal into a one-sided amplitude spectrum."""

from collections.abc import Iterable
import math
from numbers import Complex, Real

from .fourier import dft


def amplitude_spectrum(
    samples: Iterable[float], sample_rate: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (frequencies_hz, amplitudes) for a nonempty real signal.

    Use the educational O(N**2) DFT, without a window or zero padding. Keep
    bins 0 through N//2. Scale by 1/N, doubling every bin except DC and the
    Nyquist bin (which exists only for even N). Amplitude uses the input's
    units: integer PCM samples produce PCM amplitude, not normalized audio.

    Exact tone amplitudes require bin-centered sinusoids; off-bin frequencies
    leak across bins. Empty/nonfinite samples and nonpositive/nonfinite sample
    rates raise ValueError. Complex samples are unsupported. Use short inputs.
    """
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be finite and positive")
    values = tuple(samples)
    if any(isinstance(sample, Complex) and not isinstance(sample, Real) for sample in values):
        raise TypeError("Amplitude spectrum requires real samples")
    values = tuple(float(sample) for sample in values)
    coefficients = dft(values)
    size = len(values)
    frequencies = tuple(k * (sample_rate / size) for k in range(size // 2 + 1))
    amplitudes = []
    for k in range(len(frequencies)):
        amplitude = abs(coefficients[k]) / size
        if k != 0 and not (size % 2 == 0 and k == size // 2):
            amplitude *= 2
        amplitudes.append(amplitude)
    return frequencies, tuple(amplitudes)
