"""Track a synthetic tone switch, plot it, and retain an STFT comparison report."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import matplotlib

from autodj.audio import save_spectrogram
from autodj.dsp import dft, stft


ROOT = Path(__file__).resolve().parents[1]


def main():
    rate, frame_size, hop_size = 1024, 64, 32
    t = np.arange(256) / rate
    samples = np.concatenate((np.sin(2 * np.pi * 64 * t),
                              0.5 * np.sin(2 * np.pi * 192 * t)))
    result = stft(samples, rate, frame_size=frame_size, hop_size=hop_size)
    # Independent, slow reference: explicit slices, NumPy Hann, our direct DFT.
    weights = np.hanning(frame_size + 1)[:-1]
    expected = np.asarray([
        dft(samples[start:start + frame_size] * weights)[:frame_size // 2 + 1]
        for start in range(0, len(samples) - frame_size + 1, hop_size)
    ])
    error = float(np.max(np.abs(result.coefficients - expected)))
    peaks = result.frequencies[np.argmax(np.abs(result.coefficients), axis=1)]
    passed = bool(
        np.allclose(result.coefficients, expected, atol=1e-10, rtol=1e-10)
        and np.array_equal(peaks[:7], [64] * 7)
        and np.array_equal(peaks[8:], [192] * 7)
        and np.array_equal(result.times, np.arange(15) / 32)
    )
    for row, (start, peak) in enumerate(zip(result.times, peaks)):
        note = " (frame straddles the tone switch)" if row == 7 else ""
        print(f"Frame starts at {start:.5f}s: strongest bin {peak:g} Hz{note}")
    print(f"Maximum complex-coefficient error versus direct DFT: {error:.3e}")

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/stft" / run_id
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "tone_switch.npz", samples=samples, times=result.times,
             frequencies=result.frequencies, coefficients=result.coefficients,
             sample_rate=rate, frame_size=frame_size, hop_size=hop_size,
             window_sum=result.window_sum)
    plot_path = output / "spectrogram.png"
    save_spectrogram(result, plot_path, title="Tone switch at 0.25 s: 64 Hz → 192 Hz")
    sources = sorted((ROOT / "src/autodj/dsp").glob("*.py")) + [
        ROOT / "src/autodj/audio/spectrogram.py", Path(__file__),
    ]
    report = {
        "suite": "stft_tone_switch", "suite_version": 1,
        "created_at_utc": now.isoformat(), "passed": passed,
        "sample_rate": rate, "frame_size": frame_size, "hop_size": hop_size,
        "window": "periodic Hann", "time_convention": "frame start",
        "padding": False, "switch_time_seconds": 0.25,
        "coefficients_shape": list(result.coefficients.shape),
        "frame_start_times": result.times.tolist(), "peak_frequencies_hz": peaks.tolist(),
        "max_absolute_error": error, "atol": 1e-10, "rtol": 1e-10,
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "data": str((output / "tone_switch.npz").relative_to(ROOT)),
        "spectrogram": {"path": str(plot_path.relative_to(ROOT)),
                        "reference_amplitude": 1, "floor_db": -80, "ceiling_db": 0,
                        "column_width_seconds": hop_size / rate},
    }
    report_path = ROOT / "evals/results" / f"stft-{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(f"{'PASS' if passed else 'FAIL'} — report: {report_path.relative_to(ROOT)}")
    print(f"Arrays: {(output / 'tone_switch.npz').relative_to(ROOT)}")
    print(f"Spectrogram: {plot_path.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
