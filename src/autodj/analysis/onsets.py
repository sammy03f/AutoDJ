"""Spectral-flux onset strength and a small deterministic peak picker."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from autodj.dsp import StftResult


@dataclass(frozen=True)
class OnsetCandidates:
    """Selected frame indices and their supplied timestamps, in time order."""

    frame_indices: NDArray[np.int64]
    times: NDArray[np.float64]


def pick_onsets(
    strength: Iterable[float], times: Iterable[float], *, threshold: float,
    min_interval: float,
) -> OnsetCandidates:
    """Select local maxima above an absolute threshold with spacing in seconds.

    Strength and times must be finite, real, nonnegative 1-D sequences of equal
    length. Times must increase strictly. Threshold and min_interval must be
    finite and nonnegative. A peak must be strictly above threshold.

    A plateau qualifies only if both neighboring values are lower; choose its
    first index. Peaks touching either boundary are excluded. Fewer than three
    values therefore yields no candidates. Suppress nearby peaks in descending
    strength order, breaking ties by earlier index; return survivors in time order.
    Distances equal to min_interval are allowed (with 1e-12 relative tolerance
    for floating-point roundoff). This offline greedy rule is not a streaming
    detector or a globally optimal subset search; suppression costs O(P**2).

    Timestamps are copied as supplied, with no timing correction. When using
    spectral_flux, pass STFT frame-start times. Input arrays are unchanged.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    if not math.isfinite(min_interval) or min_interval < 0:
        raise ValueError("Minimum interval must be finite and nonnegative")
    values = np.asarray(tuple(strength))
    timestamps = np.asarray(tuple(times))
    if values.ndim != 1 or timestamps.ndim != 1 or values.size != timestamps.size:
        raise ValueError("Strength and times must be 1-D sequences of equal length")
    if np.iscomplexobj(values) or np.iscomplexobj(timestamps):
        raise TypeError("Strength and times must be real")
    values = values.astype(np.float64)
    timestamps = timestamps.astype(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Strength must be finite and nonnegative")
    if (not np.isfinite(timestamps).all() or np.any(timestamps < 0)
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError("Times must be finite, nonnegative, and strictly increasing")

    candidates = []
    index = 1
    while index < len(values) - 1:
        if values[index] > threshold and values[index] > values[index - 1]:
            end = index
            while end + 1 < len(values) and values[end + 1] == values[index]:
                end += 1
            if end + 1 < len(values) and values[end + 1] < values[index]:
                candidates.append(index)
            index = end + 1
        else:
            index += 1

    selected = []
    for index in sorted(candidates, key=lambda i: (-values[i], i)):
        distances = [abs(timestamps[index] - timestamps[other]) for other in selected]
        if all(distance >= min_interval or math.isclose(distance, min_interval, rel_tol=1e-12)
               for distance in distances):
            selected.append(index)
    indices = np.array(sorted(selected), dtype=np.int64)
    return OnsetCandidates(indices, timestamps[indices].copy())


def refine_onset_times(
    energy: Iterable[float], sample_rate: float, candidate_times: Iterable[float], *,
    search_radius: float, relative_threshold: float = 0.1,
) -> NDArray[np.float64]:
    """Estimate local energy-rise times around supplied candidates, in seconds.

    Search the closed interval candidate +/- search_radius, clipped to the
    signal. Before the earliest local maximum, find the first upward crossing
    of min + relative_threshold * (max - min). Require an observed value at or
    below that level in the search region; otherwise retain the candidate.
    Flat regions also retain the candidate. Crossings use sample timestamps,
    with no interpolation or smoothing-delay correction.

    Energy must be finite, nonnegative, 1-D and sampled at sample_rate. Times
    must be finite, strictly increasing and inside [0, len(energy)/sample_rate).
    Radius must be positive; threshold must lie strictly between zero and one.
    Output preserves candidate correspondence/count, but overlapping searches
    can produce equal or reordered times. No new events are detected. This
    offline heuristic can follow an unrelated rise in background audio.
    """
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be finite and positive")
    if not math.isfinite(search_radius) or search_radius <= 0:
        raise ValueError("Search radius must be finite and positive")
    if not math.isfinite(relative_threshold) or not 0 < relative_threshold < 1:
        raise ValueError("Relative threshold must lie strictly between zero and one")
    values = np.asarray(tuple(energy))
    times = np.asarray(tuple(candidate_times))
    if values.ndim != 1 or times.ndim != 1:
        raise ValueError("Energy and candidate times must be 1-D")
    if np.iscomplexobj(values) or np.iscomplexobj(times):
        raise TypeError("Energy and candidate times must be real")
    values = values.astype(np.float64)
    times = times.astype(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Energy must be finite and nonnegative")
    if (not np.isfinite(times).all() or np.any(times < 0)
            or np.any(times >= len(values) / sample_rate) or np.any(np.diff(times) <= 0)):
        raise ValueError("Candidate times must increase strictly and lie inside the signal")
    refined = times.copy()
    for index, time in enumerate(times):
        lower = math.ceil(max(0.0, time - search_radius) * sample_rate)
        upper = min(len(values), math.floor(min(len(values) / sample_rate,
                                               time + search_radius) * sample_rate) + 1)
        region = values[lower:upper]
        if region.size < 2:
            continue
        peak = int(np.argmax(region))
        floor = float(region.min())
        level = floor + relative_threshold * (float(region[peak]) - floor)
        crossings = np.flatnonzero((region[:peak] <= level) & (region[1:peak + 1] > level))
        if crossings.size:
            refined[index] = (lower + int(crossings[0]) + 1) / sample_rate
    return refined


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
