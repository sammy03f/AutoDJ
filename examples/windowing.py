"""Compare aligned/off-bin tones, with and without a periodic Hann window."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from time import perf_counter

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib

from autodj.dsp import amplitude_spectrum, hann_window


ROOT = Path(__file__).resolve().parents[1]


def main():
    rate, size = 1760, 64
    spacing = rate / size
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/windowing" / run_id
    output.mkdir(parents=True, exist_ok=False)
    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    results = []
    for row, tone in enumerate((220, 230)):
        samples = [math.sin(2 * math.pi * tone * n / rate) for n in range(size)]
        for window, label, color in (
            (None, "Rectangular", "#b35d43"), ("hann", "Hann (gain corrected)", "#197f89"),
        ):
            start = perf_counter()
            frequencies, amplitudes = amplitude_spectrum(samples, rate, window=window)
            elapsed = perf_counter() - start
            axes[row, 0].plot(frequencies, amplitudes, ".-", label=label, color=color)
            # Display floor only: do not modify the numerical spectrum.
            db = [20 * math.log10(max(a, 1e-6)) for a in amplitudes]
            axes[row, 1].plot(frequencies, db, ".-", label=label, color=color)
            peak = max(range(len(amplitudes)), key=amplitudes.__getitem__)
            # Fixed exclusion region avoids counting the wider Hann main lobe.
            far = [a for f, a in zip(frequencies, amplitudes) if abs(f - tone) > 2 * spacing]
            result = {
                "tone_hz": tone, "window": window, "peak_bin_hz": frequencies[peak],
                "peak_amplitude": amplitudes[peak], "max_far_amplitude": max(far),
                "elapsed_seconds": elapsed,
            }
            results.append(result)
            print(f"{tone} Hz, {label}: peak={amplitudes[peak]:.6f}, "
                  f"max distant amplitude={max(far):.6f}")
        for column, axis in enumerate(axes[row]):
            axis.set_title(f"{tone} Hz — {'aligned with a bin' if tone == 220 else 'between bins'}")
            axis.set_xlim(0, rate / 2)
            axis.set_ylim((0, 1.1) if column == 0 else (-100, 5))
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("Amplitude" if column == 0 else "Amplitude (dB relative to 1)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    figure.suptitle("Hann window: less distant leakage, a wider main peak\n"
                    "64 samples at 1760 Hz | 27.5 Hz per bin | Unit-amplitude tones")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()

    taper = Figure(figsize=(8, 3), layout="constrained")
    FigureCanvasAgg(taper)
    axis = taper.subplots()
    axis.plot(range(size), hann_window(size), ".-", color="#197f89")
    axis.set(xlabel="Sample index", ylabel="Weight", ylim=(0, 1.05),
             title="Periodic Hann window (64 samples)")
    axis.grid(alpha=0.2)
    taper.savefig(output / "hann_window.png", dpi=120)
    taper.clear()

    sources = sorted((ROOT / "src/autodj/dsp").glob("*.py")) + [Path(__file__)]
    report = {
        "suite": "windowing_observations", "suite_version": 1,
        "created_at_utc": now.isoformat(), "sample_rate": rate, "sample_count": size,
        "bin_spacing_hz": spacing, "far_exclusion_radius_hz": 2 * spacing,
        "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "plot": str((output / "comparison.png").relative_to(ROOT)), "cases": results,
    }
    report_path = ROOT / "evals/results" / f"windowing-{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(f"Images: {output.relative_to(ROOT)}")
    print(f"Report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
