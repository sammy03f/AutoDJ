"""Small analysis windows, implemented explicitly for learning."""

import math


def hann_window(size: int) -> tuple[float, ...]:
    """Return a periodic Hann window of positive integer length.

    For N > 1, w[n] = 0.5 - 0.5*cos(2*pi*n/N). This samples one period,
    omitting the repeated zero endpoint. It equals a symmetric Hann of
    length N+1 with its final element removed. Its mean is 0.5 for N > 1.
    A single-sample window is (1.0,), so it preserves that sample.
    """
    if type(size) is not int or size <= 0:
        raise ValueError("Window size must be a positive integer")
    if size == 1:
        return (1.0,)
    return tuple(0.5 - 0.5 * math.cos(2 * math.pi * n / size) for n in range(size))
