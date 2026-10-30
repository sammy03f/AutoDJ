"""Compare a predeclared raw median/MAD threshold across isolated and repeated events."""

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

from autodj.analysis import local_onset_threshold, robust_onset_threshold, pick_onsets, spectral_flux
from autodj.dsp import stft

if __package__:
    from . import onset_sequences as sequences
    from . import onset_sensitivity as isolated
else:
    import onset_sequences as sequences
    import onset_sensitivity as isolated

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("fixed", "original_adaptive", "selected_adaptive", "median_mad")
RADIUS, MAD_MULTIPLIER, FLOOR = 12, 4.0, 0.1
SEED_GROUPS = {"isolated_baseline": (401, 503, 607), "sequences_baseline": (701, 809),
               "isolated_fresh": (1013, 1217), "sequences_fresh": (1013, 1217)}


def analyze(samples):
    result = stft(samples, isolated.RATE, frame_size=isolated.FRAME_SIZE, hop_size=isolated.HOP_SIZE)
    strength = spectral_flux(result)
    times = result.times + isolated.FRAME_SIZE / (2 * isolated.RATE)
    thresholds = {"fixed": np.full(strength.size, isolated.THRESHOLD),
                  "original_adaptive": local_onset_threshold(strength, radius=8, std_multiplier=3, floor=FLOOR),
                  "selected_adaptive": local_onset_threshold(strength, radius=12, std_multiplier=2.5, floor=FLOOR),
                  "median_mad": robust_onset_threshold(strength, radius=RADIUS, mad_multiplier=MAD_MULTIPLIER, floor=FLOOR)}
    picks = {method: pick_onsets(strength, times, threshold=threshold, min_interval=isolated.MIN_INTERVAL)
             for method, threshold in thresholds.items()}
    return times, strength, thresholds, picks


def evaluate_signal(samples, truths, metadata):
    _, _, _, picks = analyze(samples)
    return {**metadata, "known_times": truths, "duration_seconds": len(samples) / isolated.RATE,
            "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
            "frame_indices": {method: pick.frame_indices.tolist() for method, pick in picks.items()},
            "times": {method: pick.times.tolist() for method, pick in picks.items()},
            "metrics": {method: sequences.score_sequence(pick.times.tolist(), truths) for method, pick in picks.items()}}


def summarize(cases):
    return sequences.summarize(cases, methods=METHODS)


def evaluate_group(regime, seeds):
    cases, controls = [], []
    if regime == "isolated":
        for kind in isolated.KINDS:
            for gain in isolated.GAINS:
                for noise in isolated.NOISE_STDS:
                    for seed in seeds:
                        for offset in isolated.OFFSETS:
                            samples, _ = isolated.make_noisy_signal(kind, gain, noise, seed, offset)
                            cases.append(evaluate_signal(samples, [(512 + offset) / isolated.RATE],
                                {"kind": kind, "gain": gain, "noise_std": noise, "seed": seed,
                                 "offset_samples": offset, "spacing_stress": False}))
        for noise in isolated.NOISE_STDS:
            for seed in seeds:
                samples, _ = isolated.make_noisy_signal(None, 0, noise, seed, 0)
                controls.append(evaluate_signal(samples, [], {"kind": "noise_only", "noise_std": noise, "seed": seed}))
    elif regime == "sequences":
        for kind in sequences.KINDS:
            for bpm in sequences.BPMS:
                for sub in sequences.SUBDIVISIONS:
                    for pattern in sequences.PATTERNS:
                        for gain in sequences.GAINS:
                            for noise in sequences.NOISE_STDS:
                                for seed in seeds:
                                    samples, starts, amplitudes = sequences.make_sequence(kind, bpm, sub, pattern, gain, noise, seed)
                                    cases.append(evaluate_signal(samples, (starts / isolated.RATE).tolist(),
                                        {"kind": kind, "bpm": bpm, "subdivision": sub, "pattern": pattern,
                                         "gain": gain, "noise_std": noise, "seed": seed,
                                         "hit_amplitudes": amplitudes.tolist(),
                                         "spacing_stress": 60 / (bpm * sub) < isolated.MIN_INTERVAL}))
        for bpm in sequences.BPMS:
            for sub in sequences.SUBDIVISIONS:
                for noise in sequences.NOISE_STDS:
                    for seed in seeds:
                        samples, _, _ = sequences.make_sequence("click", bpm, sub, "steady", 0, noise, seed)
                        controls.append(evaluate_signal(samples, [], {"kind": "noise_only", "bpm": bpm,
                            "subdivision": sub, "noise_std": noise, "seed": seed}))
    else:
        raise ValueError("Unknown fixture regime")
    return {"regime": regime, "seeds": seeds, "summary": summarize(cases),
            "regular_summary": summarize([c for c in cases if not c["spacing_stress"]]),
            "spacing_stress_summary": summarize([c for c in cases if c["spacing_stress"]]),
            "control_summary": summarize(controls), "combined_summary": summarize(cases + controls),
            "by_kind": {kind: summarize([c for c in cases if c["kind"] == kind]) for kind in sorted({c["kind"] for c in cases})},
            "by_noise": [{"noise_std": noise, "summary": summarize([c for c in cases if c["noise_std"] == noise])}
                         for noise in sorted({c["noise_std"] for c in cases})],
            "cases": cases, "controls": controls}


def save_plots(groups, output):
    figure = Figure(figsize=(14, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    colors = ("#b35d43", "#8b789b", "#197f89", "#c09524")
    for axis, (name, group) in zip(axes.flat, groups.items()):
        for index, method in enumerate(METHODS):
            m = group["combined_summary"]["methods"][method]
            axis.bar(np.arange(3) + (index - 1.5) * 0.2,
                     [m[key] for key in ("precision", "recall", "f1")], width=0.2, color=colors[index], label=method)
        axis.set_xticks(range(3), ["Precision", "Recall", "F1"])
        axis.set(ylim=(0, 1.05), title=name.replace('_', ' '))
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Frozen threshold comparison, including noise-only false detections\n"
                    "50 ms matching; sequence summaries include spacing stress")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()
    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, noise in zip(figure.subplots(2, 1), (0, 0.2)):
        samples, starts, _ = sequences.make_sequence("sharp_burst", 120, 4, "steady", 1, noise, 1013)
        times, strength, thresholds, picks = analyze(samples)
        axis.plot(times, strength, color="#68737d", label="Flux")
        for method, color in (("selected_adaptive", "#197f89"), ("median_mad", "#c09524")):
            axis.plot(times, thresholds[method], color=color, label=method)
            axis.scatter(picks[method].times, strength[picks[method].frame_indices], color=color, marker="x")
        for start in starts:
            axis.axvline(start / isolated.RATE, color="black", alpha=0.2, linestyle=":")
        axis.set(xlabel="Frame-center time (s)", ylabel="Flux",
                 title=f"120 BPM, four bursts/beat, noise std {noise}, fresh seed 1013; dotted = known hits")
        axis.legend(fontsize=8)
    figure.savefig(output / "thresholds.png", dpi=120)
    figure.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    groups = {name: evaluate_group(name.split('_')[0], seeds) for name, seeds in SEED_GROUPS.items()}
    passed = all(m["matched_count"] + m["missed_count"] == m["event_count"] and
                 m["matched_count"] + m["false_count"] == m["candidate_count"]
                 for group in groups.values() for c in group["cases"] + group["controls"] for m in c["metrics"].values())
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onset_robust" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plots(groups, output)
    sources = sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += [Path(__file__)] + [ROOT / "evals" / name for name in
                                 ("onset_sequences.py", "onset_sensitivity.py", "onset_timing.py")]
    status = isolated.git_output("status", "--porcelain")
    report = {"suite": "onset_robust_threshold", "suite_version": 1,
              "created_at_utc": now.isoformat(), "label": args.label, "accounting_checks_passed": passed,
              "settings": {"sample_rate": isolated.RATE, "frame_size": isolated.FRAME_SIZE, "hop_size": isolated.HOP_SIZE,
                           "fixed_threshold": 10, "original_mean_std": [8, 3], "selected_mean_std": [12, 2.5],
                           "robust_radius": RADIUS, "mad_multiplier": MAD_MULTIPLIER, "mad_scaling": "raw; no 1.4826 factor",
                           "floor": FLOOR, "min_interval_seconds": isolated.MIN_INTERVAL,
                           "match_tolerance_seconds": isolated.TOLERANCE, "seed_groups": SEED_GROUPS,
                           "timing": "frame center; no energy refinement", "noise_generator": "NumPy PCG64 standard_normal",
                           "isolated": {"sample_count": isolated.SAMPLE_COUNT, "kinds": isolated.KINDS,
                                        "gains": isolated.GAINS, "noise_stds": isolated.NOISE_STDS, "offsets": isolated.OFFSETS},
                           "sequences": {"hit_count": sequences.HIT_COUNT, "kinds": sequences.KINDS,
                                         "bpms": sequences.BPMS, "subdivisions": sequences.SUBDIVISIONS,
                                         "patterns": sequences.PATTERNS, "gains": sequences.GAINS, "noise_stds": sequences.NOISE_STDS}},
              "parameter_policy": "Radius 12, raw MAD multiplier 4 and floor .1 declared before evaluating either seed group; no tuning.",
              "scoring": "Chronological one-to-one max matches then min summed absolute error. Timing conditional on matches; combined summaries include controls.",
              "groups": groups, "python": platform.python_version(), "numpy": np.__version__,
              "matplotlib": matplotlib.__version__, "platform": platform.platform(),
              "git_commit": isolated.git_output("rev-parse", "HEAD"),
              "working_tree_dirty": bool(status) if status is not None else None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plots": [str((output / name).relative_to(ROOT)) for name in ("comparison.png", "thresholds.png")]}
    path = ROOT / "evals/results" / f"onset-robust-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for name, group in groups.items():
        print(name)
        for method, m in group["combined_summary"]["methods"].items():
            print(f"  {method}: {m['matched_count']}/{m['event_count']} matches, {m['false_count']} false including controls, F1={m['f1']:.3f}")
    print(f"Accounting checks: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plots: {output.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
