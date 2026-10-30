"""Evaluate an autocorrelation tempo baseline on clean rhythms and ambiguity cases."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np

from autodj.analysis import estimate_tempo, spectral_flux
from autodj.dsp import stft

ROOT = Path(__file__).resolve().parents[1]
RATE, FRAME_SIZE, HOP_SIZE = 2048, 128, 32
FRAME_RATE = RATE / HOP_SIZE
MIN_BPM, MAX_BPM = 60., 200.
BPMS = (60, 90, 120, 150, 180)
BEAT_COUNT = 16
RELATIVE_TOLERANCE = 0.03


def make_rhythm(kind, bpm, *, pattern="steady", gain=1., offset=0):
    """Sixteen nominal beats; equal subdivisions carry no unique metrical label.

    Events are clicks or 128 Hz damped bursts. Alternating beats have amplitudes
    1/.2. Subdivided tracks have two identical events per nominal beat. Round
    absolute sample positions individually. No onset detector uses the schedule.
    """
    if kind not in ("click", "sharp_burst") or pattern not in ("steady", "alternating", "subdivided"):
        raise ValueError("Unknown kind or pattern")
    subdivision = 2 if pattern == "subdivided" else 1
    starts = RATE // 2 + offset + np.rint(np.arange(BEAT_COUNT * subdivision) * RATE * 60 / (bpm * subdivision)).astype(int)
    size = RATE // 2 + offset + int(np.ceil(BEAT_COUNT * RATE * 60 / bpm)) + RATE // 2
    samples = np.zeros(size)
    n = np.arange(128)
    event = np.ones(1) if kind == "click" else np.cos(2 * np.pi * 128 * n / RATE) * np.exp(-n / 24)
    for i, start in enumerate(starts):
        amplitude = gain * (0.2 if pattern == "alternating" and i % 2 else 1)
        samples[start:start + len(event)] += amplitude * event
    return samples, starts


def score_bpm(estimated, target):
    """Keep octave errors distinct; no estimate stays in the denominator."""
    if estimated is None:
        return {"classification": "no_estimate", "absolute_error_bpm": None, "relative_error": None}
    error = abs(estimated - target)
    classification = "other"
    for name, reference in (("correct", target), ("half_time", target / 2), ("double_time", target * 2)):
        if abs(estimated - reference) / reference <= RELATIVE_TOLERANCE + 1e-12:
            classification = name
            break
    return {"classification": classification, "absolute_error_bpm": error, "relative_error": error / target}


def analyze(samples):
    spectrum = stft(samples, RATE, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    strength = spectral_flux(spectrum)
    return spectrum.times, strength, estimate_tempo(strength, FRAME_RATE, min_bpm=MIN_BPM, max_bpm=MAX_BPM)


def result_fields(samples, strength, result):
    return {"input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
            "strength_sha256": hashlib.sha256(strength.astype("<f8").tobytes()).hexdigest(),
            "duration_seconds": len(samples) / RATE, "strength_frame_count": len(strength),
            "strength_peak": float(strength.max()) if strength.size else 0.0,
            "strength_mean": float(strength.mean()) if strength.size else 0.0,
            "estimated_bpm": result.bpm, "selected_lag_frames": result.lag_frames,
            "autocorrelation_score": result.score, "lags": result.lags.tolist(), "scores": result.scores.tolist()}


def evaluate_case(kind, bpm, *, pattern="steady", gain=1., offset=0):
    samples, starts = make_rhythm(kind, bpm, pattern=pattern, gain=gain, offset=offset)
    _, strength, result = analyze(samples)
    repetition = bpm * (2 if pattern == "subdivided" else 1)
    return {"kind": kind, "nominal_bpm": bpm, "pattern": pattern, "gain": gain, "offset_samples": offset,
            "known_event_times": (starts / RATE).tolist(), "event_repetition_bpm": repetition,
            "event_repetition_in_search_range": MIN_BPM <= repetition <= MAX_BPM,
            **result_fields(samples, strength, result), **score_bpm(result.bpm, bpm),
            "event_rate_scoring": score_bpm(result.bpm, repetition)}


def summarize(cases):
    counts = {key: sum(c["classification"] == key for c in cases)
              for key in ("correct", "half_time", "double_time", "other", "no_estimate")}
    errors = [c["absolute_error_bpm"] for c in cases if c["absolute_error_bpm"] is not None]
    return {"case_count": len(cases), **counts,
            "accuracy": counts["correct"] / len(cases) if cases else None,
            "estimated_case_count": len(errors),
            "mean_absolute_error_bpm_when_estimated": float(np.mean(errors)) if errors else None}


def negative_control_signals():
    n = np.arange(RATE * 8)
    burst = np.zeros(n.size)
    t = np.arange(128)
    burst[RATE:RATE + 128] = np.cos(2 * np.pi * 128 * t / RATE) * np.exp(-t / 24)
    signals = {"silence": np.zeros(n.size), "steady_tone": np.sin(2 * np.pi * 128 * n / RATE),
               "single_burst": burst, "gaussian_noise": np.random.Generator(np.random.PCG64(2027)).normal(0, 0.05, n.size)}
    return signals


def negative_controls():
    controls = []
    for name, samples in negative_control_signals().items():
        _, strength, result = analyze(samples)
        controls.append({"kind": name, "tempo_returned": result.bpm is not None,
                         **result_fields(samples, strength, result)})
    return controls


def save_plots(cases, challenges, output):
    figure = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, rows, title in zip(figure.subplots(1, 2), (cases, challenges),
                                ("Steady one-event-per-beat tracks", "Alternating/subdivided ambiguity cases")):
        for pattern in sorted({c["pattern"] for c in rows}):
            subset = [c for c in rows if c["pattern"] == pattern and c["estimated_bpm"] is not None]
            axis.scatter([c["nominal_bpm"] for c in subset], [c["estimated_bpm"] for c in subset], label=pattern, alpha=0.7)
        axis.plot(BPMS, BPMS, color="black", linestyle="--", label="Nominal BPM")
        axis.plot(BPMS, np.array(BPMS) / 2, color="#68737d", linestyle=":")
        axis.plot(BPMS, np.array(BPMS) * 2, color="#68737d", linestyle=":")
        axis.set(xlabel="Nominal fixture BPM", ylabel="Estimated BPM", ylim=(25, 210), title=title)
        axis.legend(fontsize=8)
    figure.suptitle("Integer-lag autocorrelation | search 60–200 BPM | no octave correction")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()
    figure = Figure(figsize=(13, 7), layout="constrained")
    FigureCanvasAgg(figure)
    for axes, pattern in zip(figure.subplots(2, 2), ("steady", "alternating")):
        samples, _ = make_rhythm("sharp_burst", 120, pattern=pattern)
        times, strength, result = analyze(samples)
        axes[0].plot(times, strength, color="#197f89")
        axes[0].set(xlabel="Frame-start time (s)", ylabel="Raw flux", xlim=(0, 4), title=f"120 BPM: {pattern}")
        axes[1].plot(FRAME_RATE * 60 / result.lags, result.scores, marker=".", color="#7851a9")
        axes[1].axvline(120, color="black", linestyle="--", label="Nominal BPM")
        if result.bpm is not None:
            axes[1].axvline(result.bpm, color="#b35d43", label=f"Estimate {result.bpm:.2f}")
        axes[1].set(xlabel="BPM corresponding to lag", ylabel="Normalized autocorrelation", title="Score is not confidence")
        axes[1].legend(fontsize=8)
    figure.savefig(output / "autocorrelation.png", dpi=120)
    figure.clear()


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = [evaluate_case(kind, bpm, gain=gain, offset=offset) for kind in ("click", "sharp_burst")
             for bpm in BPMS for gain in (0.1, 1.) for offset in (0, 11)]
    challenges = [evaluate_case(kind, bpm, pattern=pattern, offset=11) for kind in ("click", "sharp_burst")
                  for bpm in BPMS for pattern in ("alternating", "subdivided")]
    controls = negative_controls()
    checked = cases + challenges + controls
    passed = all(np.isfinite(c["scores"]).all() and
                 (c["estimated_bpm"] is None or MIN_BPM <= c["estimated_bpm"] <= MAX_BPM) for c in checked)
    passed = passed and not controls[0]["tempo_returned"]
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/tempo" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plots(cases, challenges, output)
    sources = sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py")) + [Path(__file__)]
    status = git_output("status", "--porcelain")
    report = {"suite": "tempo_autocorrelation", "suite_version": 1, "created_at_utc": now.isoformat(),
              "label": args.label, "numerical_checks_passed": passed,
              "settings": {"sample_rate": RATE, "frame_size": FRAME_SIZE, "hop_size": HOP_SIZE,
                           "strength_frame_rate": FRAME_RATE, "min_bpm": MIN_BPM, "max_bpm": MAX_BPM,
                           "bpms": BPMS, "beat_count": BEAT_COUNT, "relative_bpm_tolerance": RELATIVE_TOLERANCE,
                           "alternate_amplitude_ratio": 0.2, "noise_control_seed": 2027,
                           "autocorrelation": "mean-subtracted; divide by zero-lag energy; no overlap correction; integer lags <= frame_count//2",
                           "primary_gains": [0.1, 1.0], "primary_offsets_samples": [0, 11]},
              "interpretation": "Nominal musical BPM is an annotation. Alternating accents/subdivisions are ambiguous. Score is not confidence; nonrhythmic audio may return a candidate. No tuning or octave correction.",
              "summary": summarize(cases), "challenge_summary": summarize(challenges),
              "by_pattern": {p: summarize([c for c in challenges if c["pattern"] == p]) for p in ("alternating", "subdivided")},
              "cases": cases, "challenges": challenges, "controls": controls,
              "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
              "platform": platform.platform(), "git_commit": git_output("rev-parse", "HEAD"),
              "working_tree_dirty": bool(status) if status is not None else None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plots": [str((output / name).relative_to(ROOT)) for name in ("comparison.png", "autocorrelation.png")]}
    path = ROOT / "evals/results" / f"tempo-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print("Primary:", report["summary"])
    print("Ambiguity cases:", report["challenge_summary"])
    print("Controls:", {c["kind"]: c["estimated_bpm"] for c in controls})
    print(f"Numerical checks: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plots: {output.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
