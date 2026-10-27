"""Plot spectral flux for known bursts; this does not select onset times."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np

from autodj.analysis import spectral_flux
from autodj.dsp import stft


ROOT = Path(__file__).resolve().parents[1]


def main():
    rate, frame_size, hop_size = 2048, 128, 32
    starts = np.array([512, 1536, 2560])
    gains = [1, 0.6, 0.8]
    burst_size = 128
    samples = np.zeros(3072)
    n = np.arange(burst_size)
    burst = np.cos(2 * np.pi * 128 * n / rate) * np.exp(-n / 24)
    for start, gain in zip(starts, gains):
        samples[start:start + burst_size] = gain * burst
    result = stft(samples, rate, frame_size=frame_size, hop_size=hop_size)
    strength = spectral_flux(result)

    # Scalar reference checks vectorization; known-answer tests check the formula.
    reference = np.zeros(len(result.times))
    for row in range(1, len(reference)):
        reference[row] = sum(
            max(0, abs(current) - abs(previous))
            for current, previous in zip(result.coefficients[row], result.coefficients[row - 1])
        )
    active = np.zeros(len(strength), dtype=bool)
    observations = []
    for start in starts:
        # Frames whose current or previous analysis window intersects this burst.
        region = ((result.times + frame_size / rate > start / rate)
                  & (result.times - hop_size / rate < (start + burst_size) / rate))
        active |= region
        observations.append({"known_start_seconds": float(start / rate),
                             "max_strength_in_known_region": float(strength[region].max())})
    error = float(np.max(np.abs(strength - reference)))
    quiet_max = float(np.max(strength[~active]))
    passed = bool(
        np.allclose(strength, reference, atol=1e-10, rtol=1e-10)
        and np.all(strength >= 0) and quiet_max <= 1e-12
        and all(item["max_strength_in_known_region"] > 0 for item in observations)
    )

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onsets" / run_id
    output.mkdir(parents=True, exist_ok=False)
    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    axes[0].plot(np.arange(len(samples)) / rate, samples, color="#197f89", linewidth=0.8)
    axes[0].set(ylabel="Sample amplitude", title="Synthetic bursts (known starts marked)")
    axes[1].plot(result.times, strength, ".-", color="#b35d43", markersize=3)
    axes[1].set(ylabel="Spectral flux (raw units)", xlabel="Time (seconds; flux uses frame starts)",
                title="Positive magnitude changes — no onset selection")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        for index, start in enumerate(starts):
            axis.axvline(start / rate, color="#68737d", linestyle="--", linewidth=0.8,
                         label="Known burst start" if index == 0 else None)
        axis.set_xlim(0, len(samples) / rate)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    figure.suptitle("Onset strength from spectral flux\n"
                    "Hann: 128 samples | Hop: 32 samples | Sample rate: 2048 Hz")
    figure.savefig(output / "onset_strength.png", dpi=120)
    figure.clear()
    np.savez(output / "bursts.npz", samples=samples, times=result.times, strength=strength,
             reference=reference, known_starts=starts / rate)

    sources = sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += sorted((ROOT / "src/autodj/analysis").glob("*.py")) + [Path(__file__)]
    report = {
        "suite": "spectral_flux_bursts", "suite_version": 1,
        "created_at_utc": now.isoformat(), "passed": passed,
        "sample_rate": rate, "frame_size": frame_size, "hop_size": hop_size,
        "time_convention": "frame start", "magnitude_scale": "raw FFT, no normalization",
        "burst_size": burst_size, "burst_frequency_hz": 128, "decay_samples": 24,
        "burst_gains": gains, "sample_count": len(samples), "frame_count": len(strength),
        "max_reference_error": error, "atol": 1e-10, "rtol": 1e-10,
        "max_silent_strength": quiet_max, "silence_tolerance": 1e-12,
        "observations": observations,
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "plot": str((output / "onset_strength.png").relative_to(ROOT)),
        "data": str((output / "bursts.npz").relative_to(ROOT)),
    }
    report_path = ROOT / "evals/results" / f"onsets-{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(f"{'PASS' if passed else 'FAIL'} — maximum reference error: {error:.3e}")
    print(f"Plot and arrays: {output.relative_to(ROOT)}")
    print(f"Report: {report_path.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
