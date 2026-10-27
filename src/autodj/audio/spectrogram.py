"""Render an existing STFT as a one-sided amplitude spectrogram."""

import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from autodj.dsp import StftResult


def _spectrogram_db(
    result: StftResult, reference_amplitude: float, floor_db: float,
) -> NDArray[np.float64]:
    """Convert raw coefficients to display dB without modifying the result."""
    if not math.isfinite(reference_amplitude) or reference_amplitude <= 0:
        raise ValueError("Reference amplitude must be finite and positive")
    if not math.isfinite(floor_db) or floor_db >= 0:
        raise ValueError("Display floor must be finite and negative")
    if result.times.size == 0:
        raise ValueError("Spectrogram requires at least one complete STFT frame")
    if result.coefficients.shape != (result.times.size, result.frame_size // 2 + 1):
        raise ValueError("STFT coefficients must have shape (frames, frequency bins)")
    if not math.isfinite(result.window_sum) or result.window_sum <= 0:
        raise ValueError("STFT window sum must be finite and positive")
    if not np.isfinite(result.coefficients).all():
        raise ValueError("STFT coefficients must be finite")

    amplitudes = np.abs(result.coefficients) / result.window_sum
    stop = -1 if result.frame_size % 2 == 0 else None
    amplitudes[:, 1:stop] *= 2
    # Avoid log(0); compute the ratio in log space to avoid dividing by a tiny reference.
    db = 20 * (np.log10(np.maximum(amplitudes, np.finfo(float).tiny))
               - math.log10(reference_amplitude))
    db[amplitudes == 0] = floor_db
    return np.clip(db, floor_db, 0)


def _build_spectrogram(
    result: StftResult, title: str, *, reference_amplitude: float = 1.0,
    floor_db: float = -80.0,
) -> Figure:
    db = _spectrogram_db(result, reference_amplitude, floor_db)
    # Each column starts at its frame timestamp and occupies one hop interval.
    time_edges = np.append(result.times, result.times[-1] + result.hop_size / result.sample_rate)
    # Frequency cells meet halfway between bins, bounded by DC and Nyquist.
    frequency_edges = np.concatenate((
        [0], (result.frequencies[:-1] + result.frequencies[1:]) / 2,
        [result.sample_rate / 2],
    ))
    figure = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    mesh = axis.pcolormesh(time_edges, frequency_edges, db.T, shading="flat",
                           cmap="magma", vmin=floor_db, vmax=0)
    axis.set_xlabel("Frame start time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title(
        f"Hann | Frame: {result.frame_size} samples | Hop: {result.hop_size} samples",
        fontsize=10,
    )
    axis.set_xlim(time_edges[0], time_edges[-1])
    axis.set_ylim(0, result.sample_rate / 2)
    figure.colorbar(mesh, ax=axis, label=f"Amplitude (dB relative to {reference_amplitude:g})")
    figure.suptitle(title)
    return figure


def save_spectrogram(
    result: StftResult, path: str | Path, *, title: str = "Amplitude spectrogram",
    reference_amplitude: float = 1.0, floor_db: float = -80.0,
) -> None:
    """Save a PNG from stft() output, preserving its coefficients.

    Correct Hann gain, double interior bins, then display 20*log10(A/reference).
    Color limits are fixed at [floor_db, 0]; values outside them saturate.
    Use reference_amplitude=32768 for raw 16-bit PCM, or 1 for unit-scale audio.
    This is amplitude, not power spectral density or measured acoustic loudness.

    Columns occupy one hop beginning at each frame's start time, not its full
    analysis window. The right plot edge is last_start + hop/sample_rate, not
    necessarily the end of the audio. Empty STFTs raise ValueError. The parent
    directory must exist; existing PNGs are overwritten. No GUI is opened.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Spectrogram output path must end in .png")
    figure = _build_spectrogram(result, title, reference_amplitude=reference_amplitude,
                                floor_db=floor_db)
    try:
        figure.savefig(path, format="png", dpi=120)
    finally:
        figure.clear()
