"""Short-time Fourier analysis of one real audio channel."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .windows import hann_window


@dataclass(frozen=True)
class StftResult:
    """Unnormalized complex coefficients, shaped (frames, frequency bins).

    times contains frame START times in seconds, not centers. frequencies is
    in Hz. Divide magnitudes by window_sum and double interior frequency bins
    to obtain one-sided amplitudes; do not double DC or even-N Nyquist.
    Arrays are ordinary NumPy arrays; frozen fields do not make them immutable.
    """

    times: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    sample_rate: float
    frame_size: int
    hop_size: int
    window_sum: float


def stft(
    samples: Iterable[float], sample_rate: float, *, frame_size: int, hop_size: int,
) -> StftResult:
    """Analyze complete frames using a periodic Hann and NumPy's real FFT.

    Require finite real 1-D samples, a finite positive sample rate, and integer
    sizes with 1 <= hop_size <= frame_size. Frame starts are 0, hop_size, ... .
    No centering, zero padding, or detrending is performed. Any incomplete
    tail is omitted. Empty/short input yields shape (0, frame_size//2 + 1).

    Coefficients retain phase and are neither amplitude-normalized nor doubled.
    The input is unchanged. Input and output reside in memory; this is not a
    streaming interface. Each frame costs O(frame_size * log(frame_size)).
    """
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be finite and positive")
    if type(frame_size) is not int or frame_size <= 0:
        raise ValueError("Frame size must be a positive integer")
    if type(hop_size) is not int or not 1 <= hop_size <= frame_size:
        raise ValueError("Hop size must be an integer from 1 through frame size")

    values = np.asarray(tuple(samples))
    if values.ndim != 1:
        raise ValueError("STFT requires a one-dimensional single-channel signal")
    if np.iscomplexobj(values):
        raise TypeError("STFT requires real samples")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("STFT samples must be finite")

    weights = np.asarray(hann_window(frame_size))
    frame_count = max(0, 1 + (len(values) - frame_size) // hop_size)
    starts = np.arange(frame_count) * hop_size
    coefficients = np.empty((frame_count, frame_size // 2 + 1), dtype=np.complex128)
    for row, start in enumerate(starts):
        coefficients[row] = np.fft.rfft(values[start:start + frame_size] * weights)

    return StftResult(
        times=starts / sample_rate,
        frequencies=np.arange(frame_size // 2 + 1) * (sample_rate / frame_size),
        coefficients=coefficients, sample_rate=float(sample_rate),
        frame_size=frame_size, hop_size=hop_size, window_sum=float(weights.sum()),
    )
