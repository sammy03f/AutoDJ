"""An inspectable autocorrelation baseline for global tempo candidates."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TempoEstimate:
    """A tempo candidate plus the searched lag curve; score is not confidence.

    Lags are integer onset-strength frames in ascending order. Scores are
    mean-subtracted autocorrelation divided by zero-lag energy. None fields
    mean no estimate; arrays may still describe the searched interval. Frozen
    fields do not make the NumPy arrays immutable.
    """

    bpm: float | None
    lag_frames: int | None
    score: float | None
    lags: NDArray[np.int64]
    scores: NDArray[np.float64]


def estimate_tempo(
    strength: Iterable[float], frame_rate: float, *, min_bpm: float = 60.0,
    max_bpm: float = 200.0,
) -> TempoEstimate:
    """Estimate a global repetition rate from uniformly sampled onset strength.

    frame_rate is the strength sampling rate (audio sample rate / STFT hop),
    not the audio sample rate. Require finite, nonnegative, real 1-D strength
    and finite positive rate/bounds with min_bpm < max_bpm.

    Scale by the maximum to avoid overflow, subtract the mean, then compute
    sum(y[:-lag] * y[lag:]) / sum(y*y). Scores are not divided by the overlap
    length, so longer lags have fewer contributing pairs. Search integer lags
    whose BPM = 60*frame_rate/lag lies inside the inclusive bounds and whose
    lag is at most len(strength)//2 (at least two period lengths of data).
    Pick the largest strictly positive score; exact ties favor the shorter
    lag/faster BPM. No interpolation, tempo prior, smoothing, peak picking,
    octave correction, or minimum-confidence gate is applied.

    Empty/constant input, no supported lag, or no positive score returns None.
    Noise and isolated changes can still produce a candidate: this is not a
    rhythm-presence classifier, beat tracker, or streaming estimator. Cost is
    O(number_of_frames * number_of_searched_lags), using NumPy dot products.
    """
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("Frame rate must be finite and positive")
    if (not math.isfinite(min_bpm) or not math.isfinite(max_bpm)
            or not 0 < min_bpm < max_bpm):
        raise ValueError("BPM bounds must be finite with 0 < min_bpm < max_bpm")
    values = np.asarray(tuple(strength))
    if values.ndim != 1:
        raise ValueError("Strength must be one-dimensional")
    if np.iscomplexobj(values):
        raise TypeError("Strength must be real")
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Strength must be finite and nonnegative")
    lags = np.arange(1, len(values) // 2 + 1, dtype=np.int64)
    with np.errstate(over="ignore"):
        bpms = (frame_rate / lags) * 60
    lags = lags[(bpms >= min_bpm) & (bpms <= max_bpm)]
    scores = np.zeros(lags.size, dtype=np.float64)
    empty = TempoEstimate(None, None, None, lags, scores)
    if not lags.size or not values.size or values.max() == 0:
        return empty
    centered = values / values.max()
    centered -= centered.mean()
    energy = float(np.dot(centered, centered))
    if energy == 0:
        return empty
    for index, lag in enumerate(lags):
        scores[index] = np.dot(centered[:-lag], centered[lag:]) / energy
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        return empty
    lag = int(lags[best])
    return TempoEstimate(float((frame_rate / lag) * 60), lag, float(scores[best]), lags, scores)
