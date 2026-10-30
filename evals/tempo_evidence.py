"""Measure optional tempo-abstention heuristics without tuning or changing defaults."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np

from autodj.analysis import estimate_tempo, spectral_flux
from autodj.dsp import stft

if __package__:
    from . import tempo as baseline
else:
    import tempo as baseline

ROOT = Path(__file__).resolve().parents[1]
HOP_SIZE = 16
PEAK_FLOOR, CORRELATION_FLOOR = 0.001, 0.2
GATES = {"baseline": (0., 0.), "strength_only": (PEAK_FLOOR, 0.),
         "correlation_only": (0., CORRELATION_FLOOR), "both": (PEAK_FLOOR, CORRELATION_FLOOR)}
SEEDS = (2027, 3011, 4001)  # Existing white-noise seed plus two fresh draws.
DURATIONS = (2, 8, 16)


def evaluate(samples, metadata):
    spectrum = stft(samples, baseline.RATE, frame_size=baseline.FRAME_SIZE, hop_size=HOP_SIZE)
    strength = spectral_flux(spectrum)
    results = {}
    for name, (peak, correlation) in GATES.items():
        result = estimate_tempo(strength, baseline.RATE / HOP_SIZE,
                                min_bpm=baseline.MIN_BPM, max_bpm=baseline.MAX_BPM,
                                min_peak_strength=peak, min_correlation=correlation)
        row = {"estimated_bpm": result.bpm, "selected_lag_frames": result.lag_frames,
               "score": result.score, "rejection_reason": result.rejection_reason,
               "accepted": result.bpm is not None}
        if "nominal_bpm" in metadata:
            row.update(baseline.score_bpm(result.bpm, metadata["nominal_bpm"]))
        results[name] = row
        if name == "baseline":
            lags, scores = result.lags.tolist(), result.scores.tolist()
    return {**metadata, "duration_seconds": len(samples) / baseline.RATE,
            "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
            "strength_sha256": hashlib.sha256(strength.astype("<f8").tobytes()).hexdigest(),
            "strength_peak": float(strength.max()) if strength.size else 0.,
            "strength_mean": float(strength.mean()) if strength.size else 0.,
            "strength_frame_count": len(strength), "baseline_lags": lags, "baseline_scores": scores,
            "results": results}


def rhythm_cases():
    cases = []
    for group, gains in (("clean", (0.1, 1.)), ("quiet_stress", (1e-6, 1e-4))):
        for kind in ("click", "sharp_burst"):
            for bpm in baseline.BPMS:
                for gain in gains:
                    for offset in (0, 11):
                        samples, _ = baseline.make_rhythm(kind, bpm, gain=gain, offset=offset)
                        cases.append(evaluate(samples, {"group": group, "kind": kind, "nominal_bpm": bpm,
                            "gain": gain, "pattern": "steady", "offset_samples": offset}))
    for kind in ("click", "sharp_burst"):
        for bpm in baseline.BPMS:
            for pattern in ("alternating", "subdivided"):
                samples, _ = baseline.make_rhythm(kind, bpm, pattern=pattern, offset=11)
                cases.append(evaluate(samples, {"group": "ambiguity", "kind": kind, "nominal_bpm": bpm,
                    "pattern": pattern, "gain": 1., "offset_samples": 11}))
    for gain in (0.1, 1.):
        for std in (0.05, 0.2):
            for seed in SEEDS[1:]:
                samples, _ = baseline.make_rhythm("sharp_burst", 120, gain=gain, offset=11)
                noise = np.random.Generator(np.random.PCG64(seed)).normal(0, std, samples.size)
                cases.append(evaluate(samples + noise, {"group": "noisy_rhythms", "kind": "sharp_burst",
                    "nominal_bpm": 120, "gain": gain, "noise_std": std, "seed": seed,
                    "pattern": "steady", "offset_samples": 11}))
    return cases


def control_signals():
    """Controls have no declared stable beat; random bursts may coincide by chance."""
    rows = []
    for duration in DURATIONS:
        n = np.arange(baseline.RATE * duration)
        rows.append((np.zeros(n.size), {"kind": "silence"}))
        single = np.zeros(n.size)
        t = np.arange(128)
        event = np.cos(2 * np.pi * 128 * t / baseline.RATE) * np.exp(-t / 24)
        single[baseline.RATE:baseline.RATE + len(event)] = event
        rows.append((single, {"kind": "single_burst"}))
        for frequency in (128, 137):
            for gain in (0.01, 1.):
                samples = gain * np.sin(2 * np.pi * frequency * n / baseline.RATE)
                rows.append((samples, {"kind": "steady_tone", "frequency_hz": frequency, "gain": gain}))
        for seed in SEEDS:
            rng = np.random.Generator(np.random.PCG64(seed))
            unit_noise = rng.standard_normal(n.size)
            for std in (0.01, 0.2):
                rows.append((unit_noise * std, {"kind": "white_noise", "noise_std": std, "seed": seed}))
            # Sparse irregular attacks over the duration, never supplied as tempo truth.
            rng = np.random.Generator(np.random.PCG64(seed))
            starts = np.sort(rng.choice(np.arange(256, n.size - 256), size=duration * 2, replace=False))
            samples = np.zeros(n.size)
            for start in starts:
                samples[start:start + len(event)] += event
            rows.append((samples, {"kind": "irregular_bursts", "seed": seed,
                                   "event_times": (starts / baseline.RATE).tolist()}))
    return rows


def summarize(cases):
    summary = {}
    for method in GATES:
        rows = [case["results"][method] for case in cases]
        accepted = sum(row["accepted"] for row in rows)
        labeled = [r for r in rows if "classification" in r]
        correct = sum(r["classification"] == "correct" for r in labeled)
        summary[method] = {"case_count": len(rows), "accepted": accepted, "abstained": len(rows) - accepted,
                           "acceptance_rate": accepted / len(rows) if rows else None,
                           "nominal_bpm_case_count": len(labeled), "correct": correct,
                           "nominal_accuracy": correct / len(labeled) if labeled else None,
                           "classification_counts": dict(Counter(r["classification"] for r in labeled)),
                           "rejection_counts": dict(Counter(r["rejection_reason"] for r in rows if not r["accepted"]))}
    return summary


def check_case(case):
    original = case["results"]["baseline"]
    return all((not r["accepted"] and r["rejection_reason"] is not None and r["score"] is None) or
               (r["accepted"] and original["accepted"] and r["estimated_bpm"] == original["estimated_bpm"]
                and r["score"] == original["score"] and r["rejection_reason"] is None)
               for r in case["results"].values())


def save_plot(cases, controls, output):
    figure = Figure(figsize=(14, 6), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    groups = ("clean", "quiet_stress", "noisy_rhythms", "ambiguity")
    for i, method in enumerate(GATES):
        left.bar(np.arange(len(groups)) + (i - 1.5) * 0.2,
                 [summarize([c for c in cases if c["group"] == group])[method]["nominal_accuracy"] for group in groups],
                 width=0.2, label=method)
    left.set_xticks(range(len(groups)), [g.replace('_', '\n') for g in groups])
    left.set(ylabel="Nominal BPM accuracy within 3% (abstentions count as misses)", ylim=(0, 1.1))
    left.legend(fontsize=8)
    kinds = ("silence", "single_burst", "steady_tone", "white_noise", "irregular_bursts")
    for i, method in enumerate(GATES):
        right.bar(np.arange(len(kinds)) + (i - 1.5) * 0.2,
                  [summarize([c for c in controls if c["kind"] == kind])[method]["acceptance_rate"] for kind in kinds],
                  width=0.2)
    right.set_xticks(range(len(kinds)), [k.replace('_', '\n') for k in kinds], fontsize=8)
    right.set(ylabel="Fraction of controls returning a tempo", ylim=(0, 1.1))
    figure.suptitle("Optional abstention checks: raw peak >= .001 and correlation >= .2\n"
                    "Frame 128 / hop 16 | Frozen settings, not calibrated confidence")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = rhythm_cases()
    controls = [evaluate(samples, {"group": "control", **metadata}) for samples, metadata in control_signals()]
    passed = all(check_case(c) for c in cases + controls)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/tempo_evidence" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plot(cases, controls, output)
    sources = sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += [Path(__file__), ROOT / "evals/tempo.py"]
    status = baseline.git_output("status", "--porcelain")
    report = {"suite": "tempo_evidence", "suite_version": 1, "created_at_utc": now.isoformat(),
              "label": args.label, "consistency_checks_passed": passed,
              "settings": {"sample_rate": baseline.RATE, "frame_size": baseline.FRAME_SIZE, "hop_size": HOP_SIZE,
                           "min_bpm": baseline.MIN_BPM, "max_bpm": baseline.MAX_BPM,
                           "min_peak_strength": PEAK_FLOOR, "min_correlation": CORRELATION_FLOOR,
                           "gates": GATES, "noise_seeds": SEEDS, "control_durations_seconds": DURATIONS,
                           "relative_bpm_tolerance": baseline.RELATIVE_TOLERANCE,
                           "beat_count": baseline.BEAT_COUNT, "bpms": baseline.BPMS,
                           "clean_gains": [0.1, 1.], "quiet_stress_gains": [1e-6, 1e-4],
                           "offset_samples": [0, 11], "tone_frequencies_hz": [128, 137],
                           "tone_gains": [0.01, 1.], "white_noise_stds": [0.01, 0.2],
                           "noisy_rhythm_stds": [0.05, 0.2], "noise_generator": "NumPy PCG64 standard_normal",
                           "alternating_gain_ratio": 0.2},
              "parameter_policy": "Peak .001 and correlation .2 declared before evaluation, no tuning; hop 16 from earlier diagnostic. Default gates remain disabled.",
              "interpretation": "Score is not confidence; absolute peak gate depends on input gain and STFT. Tone modulation/artifacts and irregular/noisy inputs may still return candidates. Controls have no declared stable tempo, not a perceptual ground-truth claim.",
              "by_group": {g: summarize([c for c in cases if c["group"] == g]) for g in ("clean", "quiet_stress", "ambiguity", "noisy_rhythms")},
              "control_summary": summarize(controls),
              "by_control_kind": {k: summarize([c for c in controls if c["kind"] == k]) for k in sorted({c["kind"] for c in controls})},
              "by_control_duration": [{"duration_seconds": d, "summary": summarize([c for c in controls if c["duration_seconds"] == d])} for d in DURATIONS],
              "cases": cases, "controls": controls,
              "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
              "platform": platform.platform(), "git_commit": baseline.git_output("rev-parse", "HEAD"),
              "working_tree_dirty": bool(status) if status is not None else None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plot": str((output / "comparison.png").relative_to(ROOT))}
    path = ROOT / "evals/results" / f"tempo-evidence-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for name, summary in report["by_group"].items():
        print(name, {method: {k: m[k] for k in ("case_count", "accepted", "correct")} for method, m in summary.items()})
    print("Controls:", {method: {k: m[k] for k in ("case_count", "accepted", "abstained")} for method, m in report["control_summary"].items()})
    print(f"Consistency checks: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plot: {(output / 'comparison.png').relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
