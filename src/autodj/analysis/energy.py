"""Short-window energy for inspecting changes in a mono signal."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def short_time_energy(samples: Iterable[float], *, window_size: int) -> NDArray[np.float64]:
    """Return trailing mean-square energy at every sample (no decimation).

    E[n] = sum(x[n-j]**2 for j in range(window_size)) / window_size.
    Samples before index zero are zero. Thus the output has the input length,
    and cannot rise before a sound starts. This is energy, not RMS or loudness.
    Direct convolution costs O(sample_count * window_size); use short windows.
    """
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError("Window size must be an integer")
    if window_size < 1:
        raise ValueError("Window size must be positive")
    values = np.asarray(tuple(samples))
    if values.ndim != 1:
        raise ValueError("Samples must be a 1-D sequence")
    if np.iscomplexobj(values):
        raise TypeError("Samples must be real")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Samples must be finite")
    if not values.size:
        return np.empty(0, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        power = values * values
        energy = np.convolve(power, np.ones(window_size) / window_size)[:values.size]
    if not np.isfinite(energy).all():
        raise ValueError("Sample magnitudes are too large to compute finite energy")
    return energy
