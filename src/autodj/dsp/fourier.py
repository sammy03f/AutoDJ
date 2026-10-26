"""An educational direct implementation of the discrete Fourier transform."""

import cmath
from collections.abc import Iterable
import math


def dft(samples: Iterable[complex]) -> tuple[complex, ...]:
    """Compute the unnormalized forward DFT of a nonempty 1-D signal.

    Accept finite real or complex numbers from one channel. Return N complex
    coefficients in standard FFT order: DC, positive frequencies, then negative
    frequencies (with the Nyquist bin between them for even N). The sample
    rate is only needed to interpret bins in Hz, not to compute the transform.

    Uses X[k] = sum(x[n] * exp(-2*pi*j*k*n/N)), matching numpy.fft.fft's
    default convention. Floating-point roundoff can leave tiny residuals.
    This O(N**2) implementation is for short signals, not full music tracks.
    Empty or nonfinite inputs raise ValueError. Input samples are unchanged.
    """
    values = tuple(complex(sample) for sample in samples)
    if not values:
        raise ValueError("DFT requires at least one sample")
    if any(not cmath.isfinite(value) for value in values):
        raise ValueError("DFT samples must be finite")

    size = len(values)
    spectrum = []
    for k in range(size):
        coefficient = 0j
        for n, sample in enumerate(values):
            angle = -2 * math.pi * k * n / size
            coefficient += sample * cmath.exp(1j * angle)
        spectrum.append(coefficient)
    return tuple(spectrum)
