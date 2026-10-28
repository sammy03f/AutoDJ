"""Separate threshold and spacing effects with a frozen two-by-two experiment."""

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

from autodj.analysis import pick_onsets

if __package__:
    from . import onset_sequences as sequences
else:
    import onset_sequences as sequences

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("all_peaks", "threshold_only", "spacing_only", "threshold_and_spacing")


def diagnose(strength, times, thresholds, truths, *, min_interval=0.1) -> dict:
    """Toggle each filter independently; threshold=0 retains positive local peaks.

    Plateau/boundary rules always remain active. A no-filter match is not a
    guaranteed true onset: especially in noise it may be a chance match.
    Losses are differences in optimal one-to-one match counts along the nested
    all-peaks -> threshold-only -> full path, not causal labels for individual hits.
    """
    strength, times, thresholds = np.asarray(strength), np.asarray(times), np.asarray(thresholds)
    selections = {name: pick_onsets(strength, times, threshold=threshold, min_interval=spacing)
                  for name, threshold, spacing in (
                      ("all_peaks", 0, 0), ("threshold_only", thresholds, 0),
                      ("spacing_only", 0, min_interval),
                      ("threshold_and_spacing", thresholds, min_interval))}
    indices = {name: result.frame_indices.tolist() for name, result in selections.items()}
    metrics = {name: sequences.score_sequence(result.times.tolist(), truths)
               for name, result in selections.items()}
    raw = set(indices["all_peaks"])
    thresholded = set(indices["threshold_only"])
    final = set(indices["threshold_and_spacing"])
    rejected = sorted(raw - thresholded)
    suppressed = sorted(thresholded - final)
    count = lambda name: metrics[name]["matched_count"]
    return {
        "frame_indices": indices,
        "times": {name: result.times.tolist() for name, result in selections.items()},
        "metrics": metrics,
        "threshold_rejected_frames": rejected, "spacing_suppressed_frames": suppressed,
        "candidate_stages": {"positive_local_peaks": len(raw), "threshold_rejected": len(rejected),
                             "spacing_suppressed_after_threshold": len(suppressed), "survivors": len(final)},
        "match_losses": {"unmatched_with_all_peaks": len(truths) - count("all_peaks"),
                         "additional_misses_after_threshold": count("all_peaks") - count("threshold_only"),
                         "additional_misses_after_spacing": count("threshold_only") - count("threshold_and_spacing")},
        "spacing_match_loss_without_threshold": count("all_peaks") - count("spacing_only"),
        "peak_trace": [{"frame": i, "time": float(times[i]), "strength": float(strength[i]),
                        "threshold": float(thresholds[i]),
                        "status": "threshold_rejected" if i in rejected else
                                  "spacing_suppressed" if i in suppressed else "retained"}
                       for i in indices["all_peaks"]],
    }


def evaluate_case(kind, bpm, subdivision, pattern, gain, noise, seed, *, control=False) -> dict:
    samples, starts, amplitudes = sequences.make_sequence(kind, bpm, subdivision, pattern,
                                                         0 if control else gain, noise, seed)
    truths = [] if control else (starts / sequences.RATE).tolist()
    times, strength, thresholds, _ = sequences.analyze(samples)
    return {"kind": "noise_only" if control else kind, "bpm": bpm, "subdivision": subdivision,
            "pattern": pattern, "gain": 0 if control else gain, "noise_std": noise, "seed": seed,
            "known_times": truths, "hit_amplitudes": [] if control else amplitudes.tolist(),
            "spacing_stress": 60 / (bpm * subdivision) < sequences.baseline.MIN_INTERVAL,
            "duration_seconds": len(samples) / sequences.RATE,
            "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
            **diagnose(strength, times, thresholds["selected_adaptive"], truths,
                       min_interval=sequences.baseline.MIN_INTERVAL)}


def summarize(cases: list[dict]) -> dict:
    result = sequences.summarize(cases, methods=METHODS)
    for field, keys in (
        ("candidate_stages", ("positive_local_peaks", "threshold_rejected", "spacing_suppressed_after_threshold", "survivors")),
        ("match_losses", ("unmatched_with_all_peaks", "additional_misses_after_threshold", "additional_misses_after_spacing")),
    ):
        result[field] = {key: sum(case[field][key] for case in cases) for key in keys}
    result["spacing_match_loss_without_threshold"] = sum(c["spacing_match_loss_without_threshold"] for c in cases)
    return result


def check_case(case: dict) -> bool:
    sets = {name: set(indices) for name, indices in case["frame_indices"].items()}
    stages, losses, metrics = case["candidate_stages"], case["match_losses"], case["metrics"]
    return (sets["threshold_and_spacing"] <= sets["threshold_only"] <= sets["all_peaks"]
            and sets["spacing_only"] <= sets["all_peaks"]
            and stages["positive_local_peaks"] == stages["threshold_rejected"] +
            stages["spacing_suppressed_after_threshold"] + stages["survivors"]
            and all(value >= 0 for value in losses.values())
            and sum(losses.values()) == metrics["threshold_and_spacing"]["missed_count"]
            and all(m["matched_count"] + m["missed_count"] == m["event_count"] and
                    m["matched_count"] + m["false_count"] == m["candidate_count"] for m in metrics.values()))


def save_plots(cases: list[dict], output: Path) -> None:
    figure = Figure(figsize=(13, 6), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    groups = [summarize([c for c in cases if c["bpm"] == bpm and c["subdivision"] == sub])
              for bpm in sequences.BPMS for sub in sequences.SUBDIVISIONS]
    labels = [f"{bpm} BPM\n{sub}/beat" for bpm in sequences.BPMS for sub in sequences.SUBDIVISIONS]
    bottom = np.zeros(6)
    for key, label, color in (
        ("unmatched_with_all_peaks", "Unmatched before filters", "#68737d"),
        ("additional_misses_after_threshold", "Additional misses after threshold", "#197f89"),
        ("additional_misses_after_spacing", "Additional misses after spacing", "#b35d43"),
    ):
        values = np.array([g["match_losses"][key] for g in groups])
        left.bar(range(6), values, bottom=bottom, label=label, color=color)
        bottom += values
    left.set_xticks(range(6), labels, fontsize=8)
    left.set(ylabel="Missed hits (384 known hits per group)", title="Losses along threshold-then-spacing path")
    left.legend(fontsize=8)
    summary = summarize(cases)
    for index, key in enumerate(("matched_count", "false_count")):
        right.bar(np.arange(4) + (index - 0.5) * 0.35,
                  [summary["methods"][m][key] for m in METHODS], width=0.35, label=key.replace('_', ' '))
    right.set_xticks(range(4), ["All positive\npeaks", "Threshold\nonly", "Spacing\nonly", "Threshold +\nspacing"], fontsize=8)
    right.set(title="All event tracks: removing filters also admits noise", ylabel="Count")
    right.legend(fontsize=8)
    figure.suptitle("Selected threshold fixed at radius 12 / 2.5 std; spacing 0 or 100 ms")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()

    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, bpm in zip(figure.subplots(2, 1), (120, 180)):
        samples, starts, _ = sequences.make_sequence("sharp_burst", bpm, 4, "steady", 1, 0, 701)
        times, strength, thresholds, _ = sequences.analyze(samples)
        diagnostic = diagnose(strength, times, thresholds["selected_adaptive"], (starts / sequences.RATE).tolist())
        axis.plot(times, strength, color="#68737d", label="Flux")
        axis.plot(times, thresholds["selected_adaptive"], color="#197f89", label="Selected threshold")
        for start in starts:
            axis.axvline(start / sequences.RATE, color="black", alpha=0.2, linestyle=":")
        for status, marker, color in (("threshold_rejected", "x", "#197f89"),
                                      ("spacing_suppressed", "s", "#b35d43"), ("retained", "o", "black")):
            peaks = [p for p in diagnostic["peak_trace"] if p["status"] == status]
            axis.scatter([p["time"] for p in peaks], [p["strength"] for p in peaks],
                         marker=marker, color=color, label=status.replace('_', ' '))
        axis.set(xlabel="Frame-center time (s)", ylabel="Flux",
                 title=f"Clean steady bursts: {bpm} BPM, four hits/beat; dotted = known events")
        axis.legend(fontsize=8, ncol=3)
    figure.savefig(output / "peak-stages.png", dpi=120)
    figure.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = [evaluate_case(kind, bpm, sub, pattern, gain, noise, seed)
             for kind in sequences.KINDS for bpm in sequences.BPMS for sub in sequences.SUBDIVISIONS
             for pattern in sequences.PATTERNS for gain in sequences.GAINS
             for noise in sequences.NOISE_STDS for seed in sequences.SEEDS]
    controls = [evaluate_case("click", bpm, sub, "steady", 0, noise, seed, control=True)
                for bpm in sequences.BPMS for sub in sequences.SUBDIVISIONS
                for noise in sequences.NOISE_STDS for seed in sequences.SEEDS]
    passed = all(check_case(c) for c in cases + controls)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onset_diagnostics" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plots(cases, output)
    sources = sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += [Path(__file__)] + [ROOT / "evals" / name for name in
                                 ("onset_sequences.py", "onset_sensitivity.py", "onset_timing.py")]
    status = sequences.baseline.git_output("status", "--porcelain")
    report = {"suite": "onset_filter_diagnostics", "suite_version": 1,
              "created_at_utc": now.isoformat(), "label": args.label, "accounting_checks_passed": passed,
              "settings": {"sample_rate": sequences.RATE, "frame_size": sequences.baseline.FRAME_SIZE,
                           "hop_size": sequences.baseline.HOP_SIZE, "hit_count": sequences.HIT_COUNT,
                           "bpms": sequences.BPMS, "subdivisions": sequences.SUBDIVISIONS,
                           "patterns": sequences.PATTERNS, "gains": sequences.GAINS,
                           "noise_stds": sequences.NOISE_STDS, "seeds": sequences.SEEDS,
                           "radius": 12, "std_multiplier": 2.5, "floor": 0.1,
                           "spacing_seconds": sequences.baseline.MIN_INTERVAL,
                           "disabled_threshold": 0, "disabled_spacing_seconds": 0,
                           "match_tolerance_seconds": sequences.baseline.TOLERANCE,
                           "timing": "frame center; no refinement"},
              "interpretation": "Match-count losses along nested candidate sets, not individual causal hit labels. Independently optimal matching can reassign pairs. No-filter noise peaks may match by chance.",
              "summary": summarize(cases),
              "regular_summary": summarize([c for c in cases if not c["spacing_stress"]]),
              "spacing_stress_summary": summarize([c for c in cases if c["spacing_stress"]]),
              "control_summary": summarize(controls),
              "by_condition": [{"bpm": bpm, "subdivision": sub,
                                "summary": summarize([c for c in cases if c["bpm"] == bpm and c["subdivision"] == sub])}
                               for bpm in sequences.BPMS for sub in sequences.SUBDIVISIONS],
              "cases": cases, "controls": controls,
              "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
              "platform": platform.platform(), "git_commit": sequences.baseline.git_output("rev-parse", "HEAD"),
              "working_tree_dirty": bool(status) if status is not None else None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plots": [str((output / name).relative_to(ROOT)) for name in ("comparison.png", "peak-stages.png")]}
    path = ROOT / "evals/results" / f"onset-diagnostics-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for name in ("regular_summary", "spacing_stress_summary"):
        print(name, report[name]["match_losses"])
        print("Candidate stages:", report[name]["candidate_stages"])
    print(f"Accounting checks: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plots: {output.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
