"""A minimal spectral-flux onset-strength baseline."""

import numpy as np
from numpy.typing import NDArray

from autodj.dsp import StftResult


def spectral_flux(result: StftResult) -> NDArray[np.float64]:
    """Sum positive changes in raw, nonnegative-frequency FFT magnitudes.

    Return one value per STFT frame, aligned with result.times (frame starts).
    flux[0] is zero: no previous frame exists. Empty input gives an empty array;
    a single frame gives [0]. The input arrays are unchanged.

    flux[m] = sum_k(max(0, abs(X[m,k]) - abs(X[m-1,k])))

    Use linear magnitudes, including DC and Nyquist, without gain correction,
    bin doubling, log compression, normalization, or smoothing. Values depend
    on input gain and STFT settings; compare only with those settings held fixed.
    This is a strength curve, not onset timestamps, probabilities, or BPM.
    """
    if result.times.ndim != 1 or result.coefficients.shape != (
        result.times.size, result.frame_size // 2 + 1,
    ):
        raise ValueError("STFT coefficients must have shape (frames, frequency bins)")
    if not np.isfinite(result.coefficients).all():
        raise ValueError("STFT coefficients must be finite")

    magnitudes = np.abs(result.coefficients)
    strength = np.zeros(result.times.size, dtype=np.float64)
    strength[1:] = np.maximum(np.diff(magnitudes, axis=0), 0).sum(axis=1)
    return strength
