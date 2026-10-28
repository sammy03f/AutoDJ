"""Compare fixed and local-statistics thresholds on baseline and fresh noise seeds."""

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

from autodj.analysis import local_onset_threshold, pick_onsets, spectral_flux
from autodj.dsp import stft

if __package__:
    from . import onset_sensitivity as baseline
else:
    import onset_sensitivity as baseline

ROOT = Path(__file__).resolve().parents[1]
DETECTORS = ("fixed", "adaptive")
RADIUS, MULTIPLIER, FLOOR = 8, 3.0, 0.1
SEED_GROUPS = {"baseline": baseline.SEEDS, "fresh": (101, 211, 307)}


def analyze(samples: np.ndarray, *, radius: int = RADIUS,
            multiplier: float = MULTIPLIER) -> tuple:
    result = stft(samples, baseline.RATE, frame_size=baseline.FRAME_SIZE, hop_size=baseline.HOP_SIZE)
    strength = spectral_flux(result)
    thresholds = local_onset_threshold(strength, radius=radius, std_multiplier=multiplier, floor=FLOOR)
    centers = result.times + baseline.FRAME_SIZE / (2 * baseline.RATE)
    selections = {name: pick_onsets(strength, centers, threshold=threshold,
                                   min_interval=baseline.MIN_INTERVAL)
                  for name, threshold in (("fixed", baseline.THRESHOLD), ("adaptive", thresholds))}
    return centers, strength, thresholds, selections


def evaluate_case(kind: str | None, gain: float, noise_std: float, seed: int, offset: int, *,
                  radius: int = RADIUS, multiplier: float = MULTIPLIER) -> dict:
    samples, noise = baseline.make_noisy_signal(kind, gain, noise_std, seed, offset)
    centers, strength, thresholds, selections = analyze(samples, radius=radius, multiplier=multiplier)
    truth = None if kind is None else (512 + offset) / baseline.RATE
    return {
        "kind": kind or "noise_only", "gain": gain, "noise_std": noise_std,
        "seed": seed, "offset_samples": offset, "known_start_seconds": truth,
        "duration_seconds": baseline.SAMPLE_COUNT / baseline.RATE,
        "realized_noise_rms": float(np.sqrt(np.mean(noise ** 2))),
        "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
        "frame_center_times": centers.tolist(), "strength": strength.tolist(),
        "adaptive_threshold": thresholds.tolist(),
        "frame_indices": {name: selection.frame_indices.tolist() for name, selection in selections.items()},
        "times": {name: selection.times.tolist() for name, selection in selections.items()},
        "metrics": {name: baseline.score_times(selection.times.tolist(), truth)
                    for name, selection in selections.items()},
    }


def summarize(cases: list[dict]) -> dict:
    return baseline.summarize(cases, methods=DETECTORS)


def evaluate_group(seeds: tuple[int, ...], *, radius: int = RADIUS,
                   multiplier: float = MULTIPLIER) -> dict:
    cases = [evaluate_case(kind, gain, std, seed, offset, radius=radius, multiplier=multiplier) for kind in baseline.KINDS
             for gain in baseline.GAINS for std in baseline.NOISE_STDS
             for seed in seeds for offset in baseline.OFFSETS]
    controls = [evaluate_case(None, 0, std, seed, 0, radius=radius, multiplier=multiplier) for std in baseline.NOISE_STDS for seed in seeds]
    return {
        "seeds": seeds, "summary": summarize(cases), "control_summary": summarize(controls),
        "by_condition": [{"gain": gain, "noise_std": std,
                          "summary": summarize([c for c in cases if c["gain"] == gain and c["noise_std"] == std])}
                         for gain in baseline.GAINS for std in baseline.NOISE_STDS],
        "by_kind": {kind: summarize([c for c in cases if c["kind"] == kind]) for kind in baseline.KINDS},
        "cases": cases, "controls": controls,
    }


def save_plots(groups: dict, output: Path) -> None:
    figure = Figure(figsize=(13, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    for row, (name, group) in enumerate(groups.items()):
        for col, (key, title) in enumerate((("matched_count", "Matched events / 576"),
                                           ("missed_count", "Missed events"),
                                           ("false_count", "False candidates\non event cases"))):
            values = [group["summary"]["methods"][method][key] for method in DETECTORS]
            axis = axes[row, col]
            bars = axis.bar(DETECTORS, values, color=("#b35d43", "#197f89"))
            axis.bar_label(bars)
            axis.set(ylim=(0, max(values + [1]) * 1.18), title=f"{name.capitalize()} seeds: {title}")
    figure.suptitle("Fixed threshold 10 vs local mean + 3 standard deviations\n"
                    "Same frame-center timing and 50 ms scoring tolerance; no energy refinement")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()

    figure = Figure(figsize=(12, 10), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, (kind, gain, std) in zip(figure.subplots(3, 1),
                                       (("sharp_burst", 0.1, 0.0), ("sharp_burst", 1.0, 0.2),
                                        ("slow_attack", 1.0, 0.0))):
        case = next(c for c in groups["fresh"]["cases"] if c["kind"] == kind
                    and c["gain"] == gain and c["noise_std"] == std
                    and c["seed"] == SEED_GROUPS["fresh"][0] and c["offset_samples"] == 0)
        times = np.array(case["frame_center_times"])
        strength = np.array(case["strength"])
        axis.plot(times, strength, color="#68737d", label="Spectral flux")
        axis.plot(times, case["adaptive_threshold"], color="#197f89", label="Adaptive threshold")
        axis.axhline(baseline.THRESHOLD, color="#b35d43", linestyle=":", label="Fixed threshold")
        axis.axvline(case["known_start_seconds"], color="black", linestyle="--", label="Known onset")
        for name, marker, color in (("fixed", "o", "#b35d43"), ("adaptive", "x", "#197f89")):
            indices = case["frame_indices"][name]
            axis.scatter(times[indices], strength[indices], marker=marker, color=color, label=f"{name} picks")
        axis.set(xlabel="Frame-center time (s)", ylabel="Raw flux",
                 title=f"{kind.replace('_', ' ').capitalize()}: gain {gain}, noise std {std}, fresh seed 101")
        axis.legend(fontsize=8, ncol=3)
    figure.savefig(output / "thresholds.png", dpi=120)
    figure.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    groups = {name: evaluate_group(seeds) for name, seeds in SEED_GROUPS.items()}
    checks = all(m["matched_count"] + m["missed_count"] == m["event_count"] and
                 m["matched_count"] + m["false_count"] == m["candidate_count"]
                 for group in groups.values() for case in group["cases"] + group["controls"]
                 for m in case["metrics"].values())
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onset_adaptive" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plots(groups, output)
    sources = sorted((ROOT / "src/autodj/dsp").glob("*.py")) + sorted((ROOT / "src/autodj/analysis").glob("*.py"))
    sources += [Path(__file__), ROOT / "evals/onset_sensitivity.py", ROOT / "evals/onset_timing.py"]
    status = baseline.git_output("status", "--porcelain")
    report = {
        "suite": "onset_adaptive_threshold", "suite_version": 1, "created_at_utc": now.isoformat(),
        "label": args.label, "accounting_checks_passed": checks,
        "settings": {"sample_rate": baseline.RATE, "sample_count": baseline.SAMPLE_COUNT,
                     "frame_size": baseline.FRAME_SIZE, "hop_size": baseline.HOP_SIZE,
                     "fixed_threshold": baseline.THRESHOLD, "adaptive_radius_frames": RADIUS,
                     "adaptive_std_multiplier": MULTIPLIER, "adaptive_floor": FLOOR,
                     "std_definition": "population std; centered clipped neighborhood including current frame",
                     "min_interval_seconds": baseline.MIN_INTERVAL,
                     "match_tolerance_seconds": baseline.TOLERANCE, "timing": "frame center, no refinement",
                     "gains": baseline.GAINS, "noise_standard_deviations": baseline.NOISE_STDS,
                     "noise_generator": "NumPy PCG64 standard_normal; paired seeds across conditions",
                     "offsets_samples": baseline.OFFSETS, "seed_groups": SEED_GROUPS},
        "parameter_policy": "radius=8, multiplier=3, floor=0.1 fixed before running either group; no search/tuning in this suite",
        "scoring": "Independent nearest match per detector within inclusive 50 ms; extras false; timing conditional on matches",
        "groups": groups,
        "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "git_commit": baseline.git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "plots": [str((output / name).relative_to(ROOT)) for name in ("comparison.png", "thresholds.png")],
    }
    path = ROOT / "evals/results" / f"onset-adaptive-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for name, group in groups.items():
        for method in DETECTORS:
            m = group["summary"]["methods"][method]
            control = group["control_summary"]["methods"][method]
            print(f"{name}/{method}: {m['matched_count']}/576 matched, {m['missed_count']} missed, "
                  f"{m['false_count']} extras; controls: {control['false_count']} false candidates")
    print(f"Accounting checks: {'PASS' if checks else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plots: {output.relative_to(ROOT)}")
    return 0 if checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
