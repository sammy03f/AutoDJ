"""Compare frame-start, frame-center and energy-refined onset timing."""

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

from autodj.analysis import pick_onsets, refine_onset_times, short_time_energy, spectral_flux
from autodj.dsp import stft


ROOT = Path(__file__).resolve().parents[1]
RATE = 2048
SAMPLE_COUNT = 1536
THRESHOLD = 10.0
MIN_INTERVAL = 0.1
ENERGY_WINDOW = 8  # 3.90625 ms at RATE; trailing mean square.
ENERGY_THRESHOLD = 0.1
METHODS = ("frame_start", "frame_center", "energy_refined")
KINDS = ("click", "sharp_burst", "slow_attack")
SETTINGS = tuple((size, size // divisor) for size in (64, 128, 256) for divisor in (8, 4, 2))


def make_signal(kind: str, onset_sample: int) -> np.ndarray:
    """Place one event in silence; the declared onset is its first nonzero sample."""
    samples = np.zeros(SAMPLE_COUNT)
    if kind == "click":
        burst = np.ones(1)
    elif kind == "sharp_burst":
        n = np.arange(128)
        burst = np.cos(2 * np.pi * 128 * n / RATE) * np.exp(-n / 24)
    elif kind == "slow_attack":
        n = np.arange(192)
        envelope = np.minimum((n + 1) / 64, 1) * np.exp(-np.maximum(n - 63, 0) / 32)
        burst = np.cos(2 * np.pi * 128 * n / RATE) * envelope
    else:
        raise ValueError(f"Unknown signal kind: {kind}")
    samples[onset_sample:onset_sample + len(burst)] = burst
    return samples


def evaluate_case(kind: str, frame_size: int, hop_size: int, offset: int,
                  *, threshold: float = THRESHOLD) -> dict:
    onset_sample = 512 + offset  # 512 is divisible by every tested hop.
    samples = make_signal(kind, onset_sample)
    result = stft(samples, RATE, frame_size=frame_size, hop_size=hop_size)
    strength = spectral_flux(result)
    start = pick_onsets(strength, result.times, threshold=threshold, min_interval=MIN_INTERVAL)
    shift = frame_size / (2 * RATE)  # Midpoint of [start, start + frame_duration).
    center = pick_onsets(strength, result.times + shift,
                         threshold=threshold, min_interval=MIN_INTERVAL)
    energy = short_time_energy(samples, window_size=ENERGY_WINDOW)
    refined = refine_onset_times(energy, RATE, center.times,
                                 search_radius=frame_size / RATE,
                                 relative_threshold=ENERGY_THRESHOLD)
    errors = None
    if len(start.times) == 1 and len(center.times) == 1:
        # No tolerance gate: retain large errors rather than discarding hard cases.
        errors = {"frame_start": float(start.times[0] - onset_sample / RATE),
                  "frame_center": float(center.times[0] - onset_sample / RATE),
                  "energy_refined": float(refined[0] - onset_sample / RATE)}
    return {
        "kind": kind, "frame_size": frame_size, "hop_size": hop_size,
        "offset_within_hop_samples": offset, "known_start_seconds": onset_sample / RATE,
        "threshold": threshold, "max_strength": float(strength.max()),
        "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
        "frame_indices": start.frame_indices.tolist(), "candidate_count": len(start.times),
        "frame_start_times": start.times.tolist(), "frame_center_times": center.times.tolist(),
        "energy_refined_times": refined.tolist(),
        "refinement_search_radius_seconds": frame_size / RATE,
        "refinement_preserves_count": len(refined) == len(start.times),
        "center_shift_seconds": shift,
        "candidate_selection_unchanged": bool(np.array_equal(start.frame_indices, center.frame_indices)),
        "errors_seconds": errors,
    }


def summarize_cases(cases: list[dict]) -> dict:
    paired = [case["errors_seconds"] for case in cases if case["errors_seconds"] is not None]
    methods = {}
    for method in METHODS:
        errors = np.array([pair[method] for pair in paired])
        methods[method] = {
            "mean_signed_error_seconds": float(errors.mean()) if errors.size else None,
            "mean_absolute_error_seconds": float(np.abs(errors).mean()) if errors.size else None,
            "max_absolute_error_seconds": float(np.abs(errors).max()) if errors.size else None,
        }
    return {
        "case_count": len(cases), "paired_timing_cases": len(paired),
        "no_candidate_cases": sum(case["candidate_count"] == 0 for case in cases),
        "multiple_candidate_cases": sum(case["candidate_count"] > 1 for case in cases),
        "center_better_cases": sum(abs(p["frame_center"]) < abs(p["frame_start"]) - 1e-12 for p in paired),
        "center_worse_cases": sum(abs(p["frame_center"]) > abs(p["frame_start"]) + 1e-12 for p in paired),
        "refined_vs_center_better_cases": sum(abs(p["energy_refined"]) < abs(p["frame_center"]) - 1e-12 for p in paired),
        "refined_vs_center_worse_cases": sum(abs(p["energy_refined"]) > abs(p["frame_center"]) + 1e-12 for p in paired),
        "methods": methods,
        "within_tolerance": [
            {"tolerance_seconds": tolerance, "denominator_all_cases": len(cases),
             **{method: sum(abs(pair[method]) <= tolerance for pair in paired)
                for method in methods}}
            for tolerance in (0.005, 0.01, 0.02, 0.05)
        ],
    }


def negative_controls() -> list[dict]:
    cases = []
    for kind in ("silence", "steady_tone"):
        samples = (np.zeros(SAMPLE_COUNT) if kind == "silence" else
                   np.sin(2 * np.pi * 128 * np.arange(SAMPLE_COUNT) / RATE))
        for size, hop in SETTINGS:
            result = stft(samples, RATE, frame_size=size, hop_size=hop)
            candidates = pick_onsets(spectral_flux(result), result.times,
                                     threshold=THRESHOLD, min_interval=MIN_INTERVAL)
            refined = refine_onset_times(
                short_time_energy(samples, window_size=ENERGY_WINDOW), RATE,
                candidates.times + size / (2 * RATE), search_radius=size / RATE,
                relative_threshold=ENERGY_THRESHOLD)
            cases.append({"kind": kind, "frame_size": size, "hop_size": hop,
                          "false_candidates": len(candidates.times),
                          "refined_count": len(refined)})
    return cases


def save_comparison(cases: list[dict], groups: dict, path: Path) -> None:
    figure = Figure(figsize=(15, 5), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    colors = ("#b35d43", "#197f89", "#7851a9")
    labels = ("Frame start", "Frame center", "Energy refined")
    for index, (method, label, color) in enumerate(zip(METHODS, labels, colors)):
        mae = [groups[kind]["methods"][method]["mean_absolute_error_seconds"] for kind in KINDS]
        left.bar(np.arange(3) + (index - 1) * 0.25,
                 [value * 1000 if value is not None else np.nan for value in mae],
                 width=0.25, color=color, label=label)
        for group, kind in enumerate(KINDS):
            errors = [case["errors_seconds"][method] * 1000 for case in cases
                      if case["kind"] == kind and case["errors_seconds"] is not None]
            right.scatter(group * 3 + index + np.linspace(-0.15, 0.15, len(errors)),
                          errors, s=14, color=color, alpha=0.75)
    left.set_xticks(range(3), [f"{kind.replace('_', ' ')}\n"
                              f"{groups[kind]['paired_timing_cases']}/{groups[kind]['case_count']} cases"
                              for kind in KINDS])
    left.set(ylabel="Mean absolute error (ms)", title="Timing error when exactly one peak is selected")
    left.legend()
    right.axhline(0, color="#68737d", linewidth=0.8)
    right.set_xticks(range(9), [f"{kind.replace('_', ' ')}\n{method}"
                              for kind in KINDS for method in ("start", "center", "energy")], fontsize=8)
    right.set(ylabel="Signed error (ms): early < 0 < late", title="Individual paired timing errors")
    figure.suptitle("Onset timing: frame timestamps vs local energy rise\n"
                    "Fixed threshold 10 | Missed/multiple detections reported separately")
    figure.savefig(path, dpi=120)
    figure.clear()


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_energy_example(path: Path) -> None:
    """Show the actual crossing on a slow attack, including its timing bias."""
    case = evaluate_case("slow_attack", 128, 32, 8)
    samples = make_signal("slow_attack", 520)
    energy = short_time_energy(samples, window_size=ENERGY_WINDOW)
    center = case["frame_center_times"][0]
    radius = case["refinement_search_radius_seconds"]
    region = energy[int((center - radius) * RATE):int((center + radius) * RATE) + 1]
    level = region.min() + ENERGY_THRESHOLD * (region.max() - region.min())
    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    waveform, curve = figure.subplots(2, 1, sharex=True)
    times = np.arange(SAMPLE_COUNT) / RATE
    waveform.plot(times, samples, color="#68737d")
    waveform.set(ylabel="Amplitude", title="Slow attack: signal-based timing remains later than first sound")
    curve.plot(times, energy, color="#7851a9", label="Trailing mean-square energy (8 samples)")
    curve.axhline(level, color="#68737d", linestyle=":", label="Local 10% energy-rise threshold")
    for axis in (waveform, curve):
        axis.axvspan(center - radius, center + radius, color="#7851a9", alpha=0.07)
        for time, label, color in (
            (case["known_start_seconds"], "Known start", "black"),
            (case["frame_start_times"][0], "Frame start", "#b35d43"),
            (center, "Frame center", "#197f89"),
            (case["energy_refined_times"][0], "Energy refined", "#7851a9"),
        ):
            axis.axvline(time, color=color, linestyle="--", label=label)
    waveform.legend(loc="upper right", fontsize=8)
    curve.legend(loc="upper right", fontsize=8)
    curve.set(xlabel="Time (s)", ylabel="Mean-square energy",
              xlim=(center - radius - 0.01, center + radius + 0.01))
    figure.savefig(path, dpi=120)
    figure.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = [evaluate_case(kind, size, hop, offset)
             for kind in KINDS for size, hop in SETTINGS
             for offset in (0, hop // 4, hop // 2, hop - 1)]
    controls = negative_controls()
    groups = {kind: summarize_cases([case for case in cases if case["kind"] == kind]) for kind in KINDS}
    summary = summarize_cases(cases)
    checks_passed = (all(case["candidate_selection_unchanged"] for case in cases)
                     and all(case["refinement_preserves_count"] for case in cases)
                     and all(case["false_candidates"] == case["refined_count"] == 0 for case in controls))
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onset_timing" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_comparison(cases, groups, output / "comparison.png")
    save_energy_example(output / "energy-example.png")
    sources = sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += sorted((ROOT / "src/autodj/analysis").glob("*.py")) + [Path(__file__)]
    status = git_output("status", "--porcelain")
    report = {
        "suite": "onset_timing_comparison", "suite_version": 2,
        "created_at_utc": now.isoformat(), "label": args.label,
        "consistency_and_control_checks_passed": checks_passed,
        "sample_rate": RATE, "sample_count": SAMPLE_COUNT,
        "threshold": THRESHOLD, "min_interval_seconds": MIN_INTERVAL,
        "center_definition": "frame start + frame_size / (2 * sample_rate)",
        "energy_refinement": {"window_size_samples": ENERGY_WINDOW,
                              "relative_threshold": ENERGY_THRESHOLD,
                              "search_radius": "frame_size / sample_rate around frame center",
                              "fallback": "retain frame center when no upward crossing"},
        "timing_population": "cases with exactly one candidate; no timing-error gate",
        "summary": summary, "by_signal_kind": groups,
        "negative_controls": controls, "cases": cases,
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "plot": str((output / "comparison.png").relative_to(ROOT)),
        "energy_plot": str((output / "energy-example.png").relative_to(ROOT)),
    }
    path = ROOT / "evals/results" / f"onset-timing-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for kind, group in groups.items():
        start = group["methods"]["frame_start"]["mean_absolute_error_seconds"]
        center = group["methods"]["frame_center"]["mean_absolute_error_seconds"]
        refined = group["methods"]["energy_refined"]["mean_absolute_error_seconds"]
        print(f"{kind}: {group['paired_timing_cases']}/{group['case_count']} paired cases; "
              f"mean absolute errors (s): start={start}, center={center}, energy={refined}; "
              f"energy worse than center in {group['refined_vs_center_worse_cases']} cases")
    print(f"Consistency/control checks: {'PASS' if checks_passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plot: {(output / 'comparison.png').relative_to(ROOT)}")
    return 0 if checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
