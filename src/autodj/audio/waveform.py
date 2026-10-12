"""Plot PCM amplitude against time without opening a GUI window."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .loader import WavAudio


def _build_waveform(audio: WavAudio, title: str) -> Figure:
    if audio.sample_rate <= 0 or audio.channels < 1:
        raise ValueError("Sample rate and channel count must be positive")

    figure = Figure(figsize=(10, 2.6 * audio.channels + 0.6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(audio.channels, 1, sharex=True, squeeze=False)
    times = [n / audio.sample_rate for n in range(audio.frame_count)]
    for channel, row in enumerate(axes):
        axis = row[0]
        values = [frame[channel] for frame in audio.samples]
        axis.plot(
            times, values, color="#197f89", linewidth=1,
            marker="." if audio.frame_count <= 200 else None, markersize=3,
        )
        axis.axhline(0, color="#89939d", linewidth=0.6)
        for limit in (-32768, 32767):
            axis.axhline(limit, color="#b35d43", linestyle="--", linewidth=0.7)
        # Keep one scale across channels and gain variants; leave room at limits.
        axis.set_ylim(-36045, 36044)
        axis.set_yticks([-32768, 0, 32767])
        axis.set_ylabel("Amplitude (PCM)")
        if audio.channels == 1:
            channel_name = "Mono"
        elif audio.channels == 2:
            channel_name = ("Left", "Right")[channel]
        else:
            channel_name = f"Channel {channel + 1}"
        axis.set_title(channel_name, loc="left", fontsize=10)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        if not audio.samples:
            axis.text(0.5, 0.5, "No samples (0 seconds)", transform=axis.transAxes,
                      ha="center", va="center", backgroundcolor="white")

    axes[-1][0].set_xlabel("Time (seconds)")
    axes[-1][0].set_xlim(0, audio.duration if audio.samples else 1)
    figure.suptitle(title)
    return figure


def save_waveform(audio: WavAudio, path: str | Path, *, title: str = "Waveform") -> None:
    """Save a PNG with one plot per channel and a fixed 16-bit amplitude scale.

    Time starts at zero for the supplied audio, including extracted clips.
    Plot every sample at n / sample_rate; connecting lines are visual guides,
    not a reconstruction of the original continuous sound. No samples are
    skipped, normalized, or modified. Prefer short clips for readable detail.

    Empty audio is annotated. The parent directory must exist; existing files
    are overwritten. The path must end in .png. Rendering works without a GUI.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Waveform output path must end in .png")
    figure = _build_waveform(audio, title)
    try:
        figure.savefig(path, format="png", dpi=120)
    finally:
        figure.clear()
