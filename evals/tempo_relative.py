"""Tune relative-flux evidence on one synthetic split, then evaluate a frozen rule."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from autodj.dsp import stft
from autodj.analysis import spectral_flux

if __package__:
    from . import tempo_evidence as evidence
    from .tempo_diagnostics import relative_peak_flux
else:
    import tempo_evidence as evidence
    from tempo_diagnostics import relative_peak_flux

baseline = evidence.baseline
ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.001, 0.01, 0.1, 0.5, 1., 2., 4., 8.)
SPLITS = {
    "tuning": {"seeds": (5101, 5107), "bpms": (80, 110, 160), "frequency": 143.},
    "evaluation": {"seeds": (6203, 6211), "bpms": (85, 115, 165), "frequency": 157.},
}
DURATIONS = (2, 8)
GAINS = (1., 1e-6)
RHYTHMS = ("sharp", "slow_attack", "noisy", "tone_backed")
CONTROLS = ("silence", "single", "tone", "multitone", "chirp", "white_noise", "colored_noise", "irregular")


def make_signal(kind, duration, seed, frequency, bpm=None):
    """Floating audio; scale the entire mixture later to preserve SNR across gains."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = duration * baseline.RATE
    t = np.arange(n) / baseline.RATE
    phase = rng.uniform(0, 2 * np.pi)
    samples = np.zeros(n)
    event_t = np.arange(256) / baseline.RATE
    event = np.cos(2 * np.pi * frequency * event_t) * np.exp(-event_t / .015)
    if kind == "slow_attack":
        event *= 1 - np.exp(-event_t / .025)
    starts = np.array([], dtype=int)
    if kind in RHYTHMS:
        starts = np.rint((.25 + np.arange(int(duration * bpm / 60) + 1) * 60 / bpm) * baseline.RATE).astype(int)
        starts = starts[starts < n]
    elif kind == "single":
        starts = np.array([baseline.RATE // 2])
    elif kind == "irregular":
        starts = np.sort(rng.choice(np.arange(256, n - 256), duration * 2, replace=False))
    for start in starts:
        length = min(len(event), n - start)
        samples[start:start + length] += event[:length]
    if kind in ("tone", "tone_backed"):
        samples += np.sin(2 * np.pi * frequency * t + phase)
    elif kind == "multitone":
        samples += (np.sin(2 * np.pi * frequency * t + phase) +
                    .7 * np.sin(2 * np.pi * (frequency + 3.7) * t))
    elif kind == "chirp":
        samples += np.sin(2 * np.pi * (frequency * t + 40 * t * t / (2 * duration)) + phase)
    elif kind in ("white_noise", "noisy"):
        samples += rng.normal(0, .1, n)
    elif kind == "colored_noise":
        samples += np.convolve(rng.normal(0, .1, n), np.ones(16) / 4, mode="full")[:n]
    elif kind not in RHYTHMS + CONTROLS:
        raise ValueError("Unknown signal kind")
    return samples, (starts / baseline.RATE).tolist()


def evaluate_split(split):
    config = SPLITS[split]
    rows = []
    for kind in RHYTHMS + CONTROLS:
        for bpm in config["bpms"] if kind in RHYTHMS else (None,):
            for duration in DURATIONS:
                for seed in config["seeds"]:
                    samples, events = make_signal(kind, duration, seed, config["frequency"], bpm)
                    for gain in GAINS:
                        audio = samples * gain
                        metadata = {"split": split, "group": "rhythm" if bpm else "control", "kind": kind,
                                    "seed": seed, "gain": gain, "frequency_hz": config["frequency"],
                                    "event_times": events}
                        if bpm is not None:
                            metadata["nominal_bpm"] = bpm
                        row = evidence.evaluate(audio, metadata)
                        spectrum = stft(audio, baseline.RATE, frame_size=baseline.FRAME_SIZE, hop_size=evidence.HOP_SIZE)
                        row["relative_peak_flux"] = relative_peak_flux(spectral_flux(spectrum), spectrum.coefficients)
                        rows.append(row)
    return rows


def relative_result(row, threshold):
    """Replace the absolute floor; retain the existing correlation-only decision."""
    result = dict(row["results"]["correlation_only"])
    ratio = row["relative_peak_flux"]
    if result["accepted"] and (ratio is None or ratio < threshold):
        result.update(accepted=False, estimated_bpm=None, selected_lag_frames=None,
                      score=None, rejection_reason="weak_relative_flux")
        if "nominal_bpm" in row:
            result.update(baseline.score_bpm(None, row["nominal_bpm"]))
    return result


def metrics(rows, method="relative", threshold=None):
    results = [(row, relative_result(row, threshold) if method == "relative" else row["results"][method]) for row in rows]
    rhythms = [r for row, r in results if row["group"] == "rhythm"]
    controls = [r for row, r in results if row["group"] == "control"]
    correct = sum(r["classification"] == "correct" for r in rhythms)
    rejected = sum(not r["accepted"] for r in controls)
    accuracy = correct / len(rhythms) if rhythms else None
    rejection = rejected / len(controls) if controls else None
    return {"rhythm_count": len(rhythms), "correct": correct,
            "rhythm_accepted": sum(r["accepted"] for r in rhythms), "rhythm_accuracy": accuracy,
            "control_count": len(controls), "controls_accepted": len(controls) - rejected,
            "control_rejection_rate": rejection,
            "balanced_utility": (accuracy + rejection) / 2 if rhythms and controls else None}


def select_threshold(tuning):
    if not tuning or any(row["split"] != "tuning" for row in tuning):
        raise ValueError("Threshold selection requires only tuning cases")
    search = [{"threshold": threshold, **metrics(tuning, threshold=threshold)} for threshold in THRESHOLDS]
    # Fixed tie break: smallest threshold, avoiding unnecessary rejection.
    selected = max(search, key=lambda row: (row["balanced_utility"], -row["threshold"]))
    return selected["threshold"], search


def comparison(rows, threshold):
    return {method: metrics(rows, method, threshold) for method in ("baseline", "correlation_only", "both", "relative")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    tuning = evaluate_split("tuning")
    threshold, search = select_threshold(tuning)
    # Selection completes before generating or analyzing evaluation waveforms.
    evaluation = evaluate_split("evaluation")
    rows = tuning + evaluation
    for row in rows:
        row["results"]["relative"] = relative_result(row, threshold)
    checks = all(evidence.check_case(row) for row in rows)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/tempo_relative" / run_id
    output.mkdir(parents=True, exist_ok=False)
    summaries = {split: comparison([r for r in rows if r["split"] == split], threshold) for split in SPLITS}
    figure = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    methods = list(summaries["evaluation"])
    for ax, key, title in zip(axes, ("rhythm_accuracy", "control_rejection_rate"),
                              ("Nominal BPM accuracy (abstentions are misses)", "Controls rejected (no declared stable beat)")):
        for i, split in enumerate(SPLITS):
            ax.bar(np.arange(len(methods)) + (i - .5) * .35,
                   [summaries[split][m][key] for m in methods], width=.35, label=split)
        ax.set_xticks(range(len(methods)), methods, rotation=15)
        ax.set(title=title, ylim=(0, 1.05))
        ax.legend()
    figure.suptitle(f"Relative flux >= {threshold:g}, correlation >= .2 | synthetic parameter holdout")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()
    sources = [Path(__file__), ROOT / "evals/tempo_evidence.py", ROOT / "evals/tempo_diagnostics.py", ROOT / "evals/tempo.py"]
    sources += sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    report = {"suite": "tempo_relative", "suite_version": 1, "created_at_utc": now.isoformat(), "label": args.label,
              "consistency_checks_passed": checks, "selected_threshold": threshold, "search": search,
              "selection_policy": "Maximize mean of nominal BPM accuracy (abstentions count as misses) and control rejection rate; ties choose smallest threshold. Tuning only; evaluation generated after selection. No retuning.",
              "settings": {"splits": SPLITS, "durations": DURATIONS, "gains": GAINS, "thresholds": THRESHOLDS,
                           "sample_rate": baseline.RATE, "frame_size": baseline.FRAME_SIZE, "hop_size": evidence.HOP_SIZE,
                           "min_correlation": evidence.CORRELATION_FLOOR, "absolute_peak_floor": evidence.PEAK_FLOOR,
                           "min_bpm": baseline.MIN_BPM, "max_bpm": baseline.MAX_BPM, "bpm_tolerance": baseline.RELATIVE_TOLERANCE},
              "limitations": "Synthetic parameter/seed holdout within shared families, not unseen-family or real-music validation. Gain copies are paired, not independent. Controls have no declared stable beat; chance rhythm and multitone beating are possible. Relative metric depends on occupancy and analysis settings. No production changes.",
              "summary": summaries,
              "evaluation_by_kind": {kind: comparison([r for r in evaluation if r["kind"] == kind], threshold) for kind in RHYTHMS + CONTROLS},
              "evaluation_by_gain": [{"gain": gain, "summary": comparison([r for r in evaluation if r["gain"] == gain], threshold)} for gain in GAINS],
              "evaluation_by_duration": [{"duration": duration, "summary": comparison([r for r in evaluation if r["duration_seconds"] == duration], threshold)} for duration in DURATIONS],
              "cases": rows, "python": platform.python_version(), "numpy": np.__version__,
              "git_commit": baseline.git_output("rev-parse", "HEAD"), "working_tree_dirty": bool(baseline.git_output("status", "--porcelain")),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plot": str((output / "comparison.png").relative_to(ROOT))}
    path = ROOT / "evals/results" / f"tempo-relative-{run_id}.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"selected_threshold": threshold, "summary": summaries, "checks": checks}, indent=2))
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plot: {(output / 'comparison.png').relative_to(ROOT)}")
    return 0 if checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
