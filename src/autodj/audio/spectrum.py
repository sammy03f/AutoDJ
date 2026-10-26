"""Render a short single-channel signal's amplitude spectrum as a PNG."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from autodj.dsp import amplitude_spectrum


def _build_spectrum(
    samples: Iterable[float], sample_rate: float, title: str,
    *, window: Literal["hann"] | None = None,
) -> Figure:
    values = tuple(samples)
    frequencies, amplitudes = amplitude_spectrum(values, sample_rate, window=window)
    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    markers, stems, _ = axis.stem(frequencies, amplitudes, basefmt=" ")
    markers.set_color("#197f89")
    markers.set_markersize(4)
    stems.set_color("#197f89")
    stems.set_linewidth(1.4)
    axis.set_xlim(0, sample_rate / 2)
    axis.set_ylim(0, max(amplitudes) * 1.15 if max(amplitudes) > 0 else 1)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Amplitude (input units)")
    window_label = "Hann (gain corrected)" if window == "hann" else "Rectangular"
    axis.set_title(f"{len(values)} samples | Bin spacing: {sample_rate / len(values):g} Hz | {window_label}",
                   fontsize=10)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title)
    return figure


def save_spectrum(
    samples: Iterable[float], sample_rate: float, path: str | Path,
    *, title: str = "Amplitude spectrum", window: Literal["hann"] | None = None,
) -> None:
    """Save a one-sided amplitude spectrum PNG for a short real signal.

    Supply one channel's samples, not interleaved stereo frames. Uses our
    O(N**2) DFT; prefer tens or hundreds of samples. Optional window="hann"
    applies a periodic Hann with coherent-gain correction. Default is no taper.
    No zero padding, dB conversion, or peak normalization is applied. Height is in input units;
    the vertical display range auto-scales to fit. The parent must exist and
    existing PNGs are overwritten. Empty input raises ValueError.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Spectrum output path must end in .png")
    figure = _build_spectrum(samples, sample_rate, title, window=window)
    try:
        figure.savefig(path, format="png", dpi=120)
    finally:
        figure.clear()
