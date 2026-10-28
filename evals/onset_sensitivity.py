"""Measure fixed-threshold onset detection across event gain and Gaussian noise."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np

from autodj.analysis import pick_onsets, refine_onset_times, short_time_energy, spectral_flux
from autodj.dsp import stft

if __package__:
    from .onset_timing import KINDS, RATE, SAMPLE_COUNT, make_signal, git_output
else:
    from onset_timing import KINDS, RATE, SAMPLE_COUNT, make_signal, git_output

ROOT = Path(__file__).resolve().parents[1]
GAINS = (0.1, 0.25, 0.5, 1.0)
NOISE_STDS = (0.0, 0.01, 0.05, 0.2)
SEEDS = (7, 19, 43)
OFFSETS = (0, 8, 16, 31)
FRAME_SIZE, HOP_SIZE = 128, 32
THRESHOLD, MIN_INTERVAL, TOLERANCE = 10.0, 0.1, 0.05
ENERGY_WINDOW, ENERGY_THRESHOLD = 8, 0.1
METHODS = ("frame_start", "frame_center", "energy_refined")


def make_noisy_signal(kind: str | None, gain: float, noise_std: float,
                      seed: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Reuse each seed's noise across gains/shapes/offsets for paired comparisons.

    Noise standard deviation is an amplitude parameter, not an SNR. No peak
    normalization or clipping is applied. None means a noise-only control.
    """
    clean = np.zeros(SAMPLE_COUNT) if kind is None else make_signal(kind, 512 + offset) * gain
    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(SAMPLE_COUNT) * noise_std
    return clean + noise, noise


def score_times(times: list[float], truth: float | None, *, tolerance: float = TOLERANCE) -> dict:
    """Match at most one candidate to the single known event, nearest first.

    All other candidates are false detections, including duplicates. A candidate
    outside the inclusive tolerance is a false detection and leaves the event
    missed. For controls every candidate is false. Ties select the first index.
    """
    matched = None
    if truth is not None and times:
        nearest = min(range(len(times)), key=lambda i: abs(times[i] - truth))
        if abs(times[nearest] - truth) <= tolerance + 1e-12:
            matched = nearest
    return {
        "event_count": int(truth is not None), "candidate_count": len(times),
        "matched_count": int(matched is not None),
        "missed_count": int(truth is not None and matched is None),
        "false_count": len(times) - int(matched is not None),
        "matched_candidate_index": matched,
        "signed_error_seconds": None if matched is None else times[matched] - truth,
    }


def evaluate_case(kind: str | None, gain: float, noise_std: float, seed: int, offset: int) -> dict:
    samples, noise = make_noisy_signal(kind, gain, noise_std, seed, offset)
    analysis = stft(samples, RATE, frame_size=FRAME_SIZE, hop_size=HOP_SIZE)
    strength = spectral_flux(analysis)
    candidates = pick_onsets(strength, analysis.times, threshold=THRESHOLD, min_interval=MIN_INTERVAL)
    centers = candidates.times + FRAME_SIZE / (2 * RATE)
    refined = refine_onset_times(short_time_energy(samples, window_size=ENERGY_WINDOW), RATE,
                                 centers, search_radius=FRAME_SIZE / RATE,
                                 relative_threshold=ENERGY_THRESHOLD)
    times = dict(zip(METHODS, (candidates.times.tolist(), centers.tolist(), refined.tolist())))
    truth = None if kind is None else (512 + offset) / RATE
    return {
        "kind": kind or "noise_only", "gain": gain, "noise_std": noise_std,
        "seed": seed, "offset_samples": offset, "known_start_seconds": truth,
        "duration_seconds": SAMPLE_COUNT / RATE,
        "realized_noise_rms": float(np.sqrt(np.mean(noise ** 2))),
        "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
        "max_strength": float(strength.max()), "frame_indices": candidates.frame_indices.tolist(),
        "times": times, "metrics": {method: score_times(values, truth) for method, values in times.items()},
    }


def summarize(cases: list[dict]) -> dict:
    duration = sum(case["duration_seconds"] for case in cases)
    methods = {}
    for method in METHODS:
        rows = [case["metrics"][method] for case in cases]
        counts = {key: sum(row[key] for row in rows) for key in
                  ("event_count", "candidate_count", "matched_count", "missed_count", "false_count")}
        errors = np.array([row["signed_error_seconds"] for row in rows
                           if row["signed_error_seconds"] is not None])
        methods[method] = {
            **counts,
            "precision": counts["matched_count"] / counts["candidate_count"] if counts["candidate_count"] else None,
            "recall": counts["matched_count"] / counts["event_count"] if counts["event_count"] else None,
            "false_detections_per_second": counts["false_count"] / duration if duration else None,
            "timing_match_count": len(errors),
            "mean_signed_error_seconds": float(errors.mean()) if errors.size else None,
            "mean_absolute_error_seconds": float(np.abs(errors).mean()) if errors.size else None,
        }
    return {"case_count": len(cases), "duration_seconds": duration, "methods": methods}


def save_plot(conditions: list[dict], controls: list[dict], path: Path) -> None:
    figure = Figure(figsize=(12, 9), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for axis, method, metric, title in (
        (axes[0, 0], "frame_center", "recall", "Frame-center recall (50 ms tolerance)"),
        (axes[0, 1], "energy_refined", "recall", "Energy-refined recall (50 ms tolerance)"),
        (axes[1, 0], "frame_center", "false_count", "Frame-center false detections per event case"),
    ):
        values = np.array([condition["summary"]["methods"][method][metric] /
                           (condition["summary"]["case_count"] if metric == "false_count" else 1)
                           for condition in conditions]).reshape(len(GAINS), len(NOISE_STDS))
        artist = axis.imshow(values, origin="lower", aspect="auto", vmin=0,
                             vmax=1 if metric == "recall" else max(1, float(values.max())), cmap="viridis")
        axis.set_xticks(range(len(NOISE_STDS)), NOISE_STDS)
        axis.set_yticks(range(len(GAINS)), GAINS)
        axis.set(xlabel="Noise standard deviation", ylabel="Event gain", title=title)
        for row in range(len(GAINS)):
            for col in range(len(NOISE_STDS)):
                axis.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center",
                          color="white", bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"))
        figure.colorbar(artist, ax=axis)
    axis = axes[1, 1]
    rates = [summarize([case for case in controls if case["noise_std"] == std])
             ["methods"]["frame_center"]["false_detections_per_second"] for std in NOISE_STDS]
    axis.bar(range(len(NOISE_STDS)), rates, color="#b35d43")
    axis.set_xticks(range(len(NOISE_STDS)), NOISE_STDS)
    axis.set(xlabel="Noise standard deviation", ylabel="False detections per second",
             title="Noise-only controls (3 seeds per level)")
    figure.suptitle("Fixed threshold 10: volume and noise sensitivity\n"
                    "36 event cases per cell | Synthetic isolated events; no detector tuning")
    figure.savefig(path, dpi=120)
    figure.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = [evaluate_case(kind, gain, std, seed, offset) for kind in KINDS
             for gain in GAINS for std in NOISE_STDS for seed in SEEDS for offset in OFFSETS]
    controls = [evaluate_case(None, 0, std, seed, 0) for std in NOISE_STDS for seed in SEEDS]
    conditions = [{"gain": gain, "noise_std": std,
                   "summary": summarize([case for case in cases if case["gain"] == gain and case["noise_std"] == std])}
                  for gain in GAINS for std in NOISE_STDS]
    # These check accounting, not detection quality; expected failures remain in the report.
    checks_passed = all(
        all(len(case["times"][method]) == len(case["frame_indices"]) and
            case["metrics"][method]["matched_count"] + case["metrics"][method]["missed_count"] ==
            case["metrics"][method]["event_count"] and
            case["metrics"][method]["matched_count"] + case["metrics"][method]["false_count"] ==
            case["metrics"][method]["candidate_count"] for method in METHODS)
        for case in cases + controls)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onset_sensitivity" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plot(conditions, controls, output / "comparison.png")
    sources = sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += sorted((ROOT / "src/autodj/analysis").glob("*.py"))
    sources += [Path(__file__), ROOT / "evals/onset_timing.py"]
    status = git_output("status", "--porcelain")
    report = {
        "suite": "onset_sensitivity", "suite_version": 1, "created_at_utc": now.isoformat(),
        "label": args.label, "accounting_checks_passed": checks_passed,
        "settings": {"sample_rate": RATE, "sample_count": SAMPLE_COUNT,
                     "frame_size": FRAME_SIZE, "hop_size": HOP_SIZE,
                     "threshold": THRESHOLD, "min_interval_seconds": MIN_INTERVAL,
                     "match_tolerance_seconds": TOLERANCE, "gains": GAINS,
                     "noise_standard_deviations": NOISE_STDS, "noise_seeds": SEEDS,
                     "noise_generator": "NumPy PCG64 standard_normal; paired seeds across conditions",
                     "offsets_samples": OFFSETS, "energy_window_samples": ENERGY_WINDOW,
                     "energy_relative_threshold": ENERGY_THRESHOLD,
                     "refinement_search_radius_seconds": FRAME_SIZE / RATE},
        "scoring": "Each method independently matches its nearest candidate within inclusive tolerance; extras are false. Timing errors are conditional on matches. Controls have no truth.",
        "summary": summarize(cases), "control_summary": summarize(controls),
        "by_condition": conditions,
        "by_kind": {kind: summarize([case for case in cases if case["kind"] == kind]) for kind in KINDS},
        "cases": cases, "controls": controls,
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "plot": str((output / "comparison.png").relative_to(ROOT)),
    }
    path = ROOT / "evals/results" / f"onset-sensitivity-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for method, metrics in report["summary"]["methods"].items():
        print(f"{method}: {metrics['matched_count']}/{metrics['event_count']} matched; "
              f"{metrics['missed_count']} missed; {metrics['false_count']} false detections")
    print(f"Accounting checks: {'PASS' if checks_passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plot: {(output / 'comparison.png').relative_to(ROOT)}")
    return 0 if checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
