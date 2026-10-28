"""Evaluate frozen onset thresholds on repeated events; BPM is fixture input only."""

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
RATE = baseline.RATE
BPMS = (60, 120, 180)
SUBDIVISIONS = (1, 4)
KINDS = ("click", "sharp_burst")
PATTERNS = ("steady", "alternating")
GAINS = (0.1, 1.0)
NOISE_STDS = (0.0, 0.05, 0.2)
SEEDS = (701, 809)
HIT_COUNT = 8
METHODS = ("fixed", "original_adaptive", "selected_adaptive")
CONFIGS = {"original_adaptive": (8, 3.0), "selected_adaptive": (12, 2.5)}


def make_sequence(kind: str, bpm: float, subdivision: int, pattern: str,
                  gain: float, noise_std: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eight scheduled events; round each absolute position, never accumulate rounding.

    Alternate amplitudes 1/.35 when requested. Padding is .25 s before the first
    event and at least .5 s after the last start. Noise is additive PCG64 Gaussian,
    with no clipping or normalization. Truth follows the clean event schedule.
    """
    if kind not in KINDS or pattern not in PATTERNS:
        raise ValueError("Unknown event kind or amplitude pattern")
    if not np.isfinite(bpm) or bpm <= 0 or subdivision < 1:
        raise ValueError("Tempo and subdivision must be positive")
    starts = 512 + np.rint(np.arange(HIT_COUNT) * RATE * 60 / (bpm * subdivision)).astype(np.int64)
    samples = np.zeros(int(starts[-1]) + RATE // 2)
    n = np.arange(128)
    event = np.ones(1) if kind == "click" else np.cos(2 * np.pi * 128 * n / RATE) * np.exp(-n / 24)
    amplitudes = gain * np.array([0.35 if pattern == "alternating" and i % 2 else 1.0 for i in range(HIT_COUNT)])
    for start, amplitude in zip(starts, amplitudes):
        samples[start:start + len(event)] += amplitude * event
    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(samples.size) * noise_std
    return samples + noise, starts, amplitudes


def score_sequence(times: list[float], truths: list[float], *, tolerance: float = baseline.TOLERANCE) -> dict:
    """Chronological one-to-one matching: maximize matches, then minimize total |error|.

    Dynamic programming handles overlapping tolerance windows. Candidates may
    duplicate; truth times must strictly increase. Exact ties prefer a match,
    then skipping a candidate, then skipping a truth. Inclusive tolerance has
    a 1e-12 s allowance for roundoff. Unmatched candidates/truths stay visible.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    for values, strict in ((times, False), (truths, True)):
        array = np.asarray(values)
        if (array.ndim != 1 or np.iscomplexobj(array) or not np.isfinite(array).all()
                or np.any(array < 0) or np.any(np.diff(array) <= 0 if strict else np.diff(array) < 0)):
            raise ValueError("Times must be finite, nonnegative, ordered 1-D sequences")
    n, m = len(truths), len(times)
    counts = np.zeros((n + 1, m + 1), dtype=int)
    errors = np.zeros((n + 1, m + 1))
    actions = np.zeros((n, m), dtype=int)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            options = []
            error = abs(times[j] - truths[i])
            if error <= tolerance + 1e-12:
                options.append((1 + counts[i + 1, j + 1], error + errors[i + 1, j + 1], 1))
            options.extend(((counts[i, j + 1], errors[i, j + 1], 2),
                            (counts[i + 1, j], errors[i + 1, j], 3)))
            count, cost, action = max(options, key=lambda option: (option[0], -option[1]))
            counts[i, j], errors[i, j], actions[i, j] = count, cost, action
    matches = []
    i = j = 0
    while i < n and j < m:
        action = actions[i, j]
        if action == 1:
            matches.append({"truth_index": i, "candidate_index": j,
                            "signed_error_seconds": times[j] - truths[i]})
            i, j = i + 1, j + 1
        elif action == 2:
            j += 1
        else:
            i += 1
    matched_truth = {pair["truth_index"] for pair in matches}
    matched_candidates = {pair["candidate_index"] for pair in matches}
    return {"event_count": n, "candidate_count": m, "matched_count": len(matches),
            "missed_count": n - len(matches), "false_count": m - len(matches),
            "matches": matches,
            "missed_truth_indices": [i for i in range(n) if i not in matched_truth],
            "false_candidate_indices": [j for j in range(m) if j not in matched_candidates]}


def analyze(samples: np.ndarray) -> tuple:
    result = stft(samples, RATE, frame_size=baseline.FRAME_SIZE, hop_size=baseline.HOP_SIZE)
    strength = spectral_flux(result)
    centers = result.times + baseline.FRAME_SIZE / (2 * RATE)
    thresholds = {"fixed": np.full(strength.size, baseline.THRESHOLD)}
    thresholds.update({name: local_onset_threshold(strength, radius=radius, std_multiplier=multiplier, floor=0.1)
                       for name, (radius, multiplier) in CONFIGS.items()})
    selections = {name: pick_onsets(strength, centers, threshold=threshold,
                                   min_interval=baseline.MIN_INTERVAL) for name, threshold in thresholds.items()}
    return centers, strength, thresholds, selections


def evaluate_case(kind: str, bpm: int, subdivision: int, pattern: str,
                  gain: float, noise_std: float, seed: int, *, control: bool = False) -> dict:
    samples, starts, amplitudes = make_sequence(kind, bpm, subdivision, pattern,
                                               0 if control else gain, noise_std, seed)
    truths = [] if control else (starts / RATE).tolist()
    _, _, _, selections = analyze(samples)
    return {"kind": "noise_only" if control else kind, "bpm": bpm,
            "subdivision": subdivision, "pattern": pattern, "gain": 0 if control else gain,
            "noise_std": noise_std, "seed": seed,
            "spacing_stress": 60 / (bpm * subdivision) < baseline.MIN_INTERVAL,
            "nominal_hit_interval_seconds": 60 / (bpm * subdivision),
            "known_times": truths, "hit_amplitudes": [] if control else amplitudes.tolist(),
            "duration_seconds": len(samples) / RATE,
            "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
            "frame_indices": {name: selection.frame_indices.tolist() for name, selection in selections.items()},
            "times": {name: selection.times.tolist() for name, selection in selections.items()},
            "metrics": {name: score_sequence(selection.times.tolist(), truths)
                        for name, selection in selections.items()}}


def summarize(cases: list[dict]) -> dict:
    duration = sum(case["duration_seconds"] for case in cases)
    methods = {}
    for method in METHODS:
        rows = [case["metrics"][method] for case in cases]
        counts = {key: sum(row[key] for row in rows) for key in
                  ("event_count", "candidate_count", "matched_count", "missed_count", "false_count")}
        tp, fp, fn = counts["matched_count"], counts["false_count"], counts["missed_count"]
        errors = np.array([pair["signed_error_seconds"] for row in rows for pair in row["matches"]])
        methods[method] = {**counts, "precision": tp / (tp + fp) if tp + fp else None,
                           "recall": tp / (tp + fn) if tp + fn else None,
                           "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                           "false_detections_per_second": fp / duration if duration else None,
                           "timing_match_count": len(errors),
                           "mean_absolute_error_seconds": float(np.abs(errors).mean()) if errors.size else None,
                           "mean_signed_error_seconds": float(errors.mean()) if errors.size else None,
                           "max_absolute_error_seconds": float(np.abs(errors).max()) if errors.size else None}
    return {"case_count": len(cases), "duration_seconds": duration, "methods": methods}


def save_plots(cases: list[dict], output: Path) -> None:
    figure = Figure(figsize=(13, 5), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    labels = [f"{bpm} BPM\n{subdivision}/beat" for bpm in BPMS for subdivision in SUBDIVISIONS]
    for index, method in enumerate(METHODS):
        groups = [summarize([c for c in cases if c["bpm"] == bpm and c["subdivision"] == subdivision])
                  for bpm in BPMS for subdivision in SUBDIVISIONS]
        left.bar(np.arange(6) + (index - 1) * 0.25,
                 [g["methods"][method]["recall"] for g in groups], width=0.25, label=method)
        right.bar(np.arange(6) + (index - 1) * 0.25,
                  [g["methods"][method]["false_count"] for g in groups], width=0.25)
    for axis in (left, right):
        axis.set_xticks(range(6), labels, fontsize=8)
    left.set(ylabel="Recall", ylim=(0, 1.15), title="One-to-one matches within 50 ms")
    left.legend(fontsize=8)
    right.set(ylabel="False candidates on event tracks", title="Extras across gain, pattern, and noise")
    figure.suptitle("Repeated events, frozen settings | 180 BPM × 4/beat is spacing stress")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()
    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, subdivision in zip(figure.subplots(2, 1), (1, 4)):
        samples, starts, _ = make_sequence("sharp_burst", 180, subdivision, "alternating", 1, 0, SEEDS[0])
        times, strength, thresholds, selections = analyze(samples)
        axis.plot(times, strength, color="#68737d", label="Flux")
        axis.plot(times, thresholds["selected_adaptive"], color="#197f89", label="Selected threshold")
        for start in starts:
            axis.axvline(start / RATE, color="black", alpha=0.3, linestyle=":")
        picks = selections["selected_adaptive"]
        axis.scatter(picks.times, strength[picks.frame_indices], color="#197f89", marker="x", label="Selected picks")
        axis.set(xlabel="Frame-center time (s)", ylabel="Flux",
                 title=f"180 BPM, {subdivision} hits/beat, alternating amplitude, no noise; dotted = known hits")
        axis.legend(fontsize=8)
    figure.savefig(output / "sequences.png", dpi=120)
    figure.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = [evaluate_case(kind, bpm, subdivision, pattern, gain, noise, seed)
             for kind in KINDS for bpm in BPMS for subdivision in SUBDIVISIONS
             for pattern in PATTERNS for gain in GAINS for noise in NOISE_STDS for seed in SEEDS]
    controls = [evaluate_case("click", bpm, subdivision, "steady", 0, noise, seed, control=True)
                for bpm in BPMS for subdivision in SUBDIVISIONS for noise in NOISE_STDS for seed in SEEDS]
    checks = all(m["matched_count"] + m["missed_count"] == m["event_count"] and
                 m["matched_count"] + m["false_count"] == m["candidate_count"]
                 for case in cases + controls for m in case["metrics"].values())
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onset_sequences" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plots(cases, output)
    sources = sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += [Path(__file__), ROOT / "evals/onset_sensitivity.py", ROOT / "evals/onset_timing.py"]
    status = baseline.git_output("status", "--porcelain")
    report = {"suite": "onset_sequences", "suite_version": 1, "created_at_utc": now.isoformat(),
              "label": args.label, "accounting_checks_passed": checks,
              "settings": {"sample_rate": RATE, "hit_count": HIT_COUNT, "bpms": BPMS,
                           "subdivisions": SUBDIVISIONS, "kinds": KINDS, "patterns": PATTERNS,
                           "gains": GAINS, "noise_stds": NOISE_STDS, "seeds": SEEDS,
                           "noise_generator": "NumPy PCG64 standard_normal, paired across compatible conditions",
                           "frame_size": baseline.FRAME_SIZE, "hop_size": baseline.HOP_SIZE,
                           "fixed_threshold": baseline.THRESHOLD, "adaptive_radius_multiplier": CONFIGS,
                           "adaptive_floor": 0.1, "min_interval_seconds": baseline.MIN_INTERVAL,
                           "match_tolerance_seconds": baseline.TOLERANCE, "timing": "frame center; no refinement",
                           "alternating_gain_ratio": 0.35, "initial_offset_samples": 512},
              "scoring": "Chronological one-to-one maximum-cardinality matching, then minimum summed absolute error; timing conditional on matches. BPM is fixture input, not estimated.",
              "summary": summarize(cases), "regular_summary": summarize([c for c in cases if not c["spacing_stress"]]),
              "spacing_stress_summary": summarize([c for c in cases if c["spacing_stress"]]),
              "control_summary": summarize(controls),
              "by_condition": [{"bpm": bpm, "subdivision": sub,
                                "summary": summarize([c for c in cases if c["bpm"] == bpm and c["subdivision"] == sub])}
                               for bpm in BPMS for sub in SUBDIVISIONS],
              "by_pattern": {p: summarize([c for c in cases if c["pattern"] == p]) for p in PATTERNS},
              "cases": cases, "controls": controls,
              "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
              "platform": platform.platform(), "git_commit": baseline.git_output("rev-parse", "HEAD"),
              "working_tree_dirty": bool(status) if status is not None else None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plots": [str((output / name).relative_to(ROOT)) for name in ("comparison.png", "sequences.png")]}
    path = ROOT / "evals/results" / f"onset-sequences-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for name in ("regular_summary", "spacing_stress_summary", "control_summary"):
        print(name)
        for method, m in report[name]["methods"].items():
            print(f"  {method}: {m['matched_count']}/{m['event_count']} matches, {m['missed_count']} misses, {m['false_count']} false")
    print(f"Accounting checks: {'PASS' if checks else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plots: {output.relative_to(ROOT)}")
    return 0 if checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
