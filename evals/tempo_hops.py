"""Measure tempo accuracy, lag scores and runtime across fixed STFT hop sizes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter_ns

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
HOPS = (32, 16, 8)
REPEATS = 3


def probe_lags(result, frame_rate, target_bpm):
    """Inspect floor/ceil of the ideal lag, without influencing estimation.

    Unsupported BPM targets are not clamped into the search range. Return
    only bins actually searched, preserving insufficient-duration exclusions.
    """
    ideal = 60 * frame_rate / target_bpm
    in_range = baseline.MIN_BPM <= target_bpm <= baseline.MAX_BPM
    wanted = {int(np.floor(ideal)), int(np.ceil(ideal))} if in_range else set()
    bins = [{"lag_frames": int(lag), "bpm": float(60 * frame_rate / lag), "score": float(score)}
            for lag, score in zip(result.lags, result.scores) if int(lag) in wanted]
    best = max(bins, key=lambda b: b["score"]) if bins else None
    return {"target_bpm": target_bpm, "ideal_lag_frames": ideal,
            "target_in_search_range": in_range, "bins": bins, "best_bracketing_bin": best}


def analyze_timed(samples, hop):
    start = perf_counter_ns()
    spectrum = stft(samples, baseline.RATE, frame_size=baseline.FRAME_SIZE, hop_size=hop)
    strength = spectral_flux(spectrum)
    features_done = perf_counter_ns()
    result = estimate_tempo(strength, baseline.RATE / hop, min_bpm=baseline.MIN_BPM, max_bpm=baseline.MAX_BPM)
    done = perf_counter_ns()
    timing = {"features_ms": (features_done - start) / 1e6,
              "tempo_ms": (done - features_done) / 1e6, "total_ms": (done - start) / 1e6}
    return strength, result, timing


def timing_summary(trials):
    return {key: {"median": float(np.median([t[key] for t in trials])),
                  "min": min(t[key] for t in trials), "max": max(t[key] for t in trials)}
            for key in ("features_ms", "tempo_ms", "total_ms")}


def evaluate_samples(samples, metadata, hop, *, repeats=REPEATS):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    trials = []
    first_signature = None
    for _ in range(repeats):
        strength, result, timing = analyze_timed(samples, hop)
        signature = (result.bpm, result.lag_frames, result.score, result.scores.tobytes())
        if first_signature is None:
            first_signature = signature
        elif signature != first_signature:
            raise RuntimeError("Repeated numerical results changed within one case")
        trials.append(timing)
    frame_rate = baseline.RATE / hop
    row = {**metadata, "hop_size": hop, "strength_frame_rate": frame_rate,
           "hop_duration_seconds": hop / baseline.RATE,
           **baseline.result_fields(samples, strength, result),
           "timing_trials_ms": trials, "timing_summary_ms": timing_summary(trials)}
    if "nominal_bpm" in metadata:
        target = metadata["nominal_bpm"]
        row.update(baseline.score_bpm(result.bpm, target))
        row["nominal_probe"] = probe_lags(result, frame_rate, target)
        row["half_time_probe"] = probe_lags(result, frame_rate, target / 2)
        repetition = target * (2 if metadata["pattern"] == "subdivided" else 1)
        row["event_repetition_bpm"] = repetition
        row["event_rate_scoring"] = baseline.score_bpm(result.bpm, repetition)
    else:
        row["tempo_returned"] = result.bpm is not None
    return row


def fixtures():
    rows = []
    for kind in ("click", "sharp_burst"):
        for bpm in baseline.BPMS:
            for gain in (0.1, 1.):
                for offset in (0, 11):
                    samples, _ = baseline.make_rhythm(kind, bpm, gain=gain, offset=offset)
                    rows.append((samples, {"group": "primary", "kind": kind, "nominal_bpm": bpm,
                                 "pattern": "steady", "gain": gain, "offset_samples": offset}))
            for pattern in ("alternating", "subdivided"):
                samples, _ = baseline.make_rhythm(kind, bpm, pattern=pattern, offset=11)
                rows.append((samples, {"group": "challenge", "kind": kind, "nominal_bpm": bpm,
                             "pattern": pattern, "gain": 1., "offset_samples": 11}))
    for kind, samples in baseline.negative_control_signals().items():
        rows.append((samples, {"group": "control", "kind": kind}))
    return rows


def run_comparison():
    inputs = fixtures()
    # One unmeasured warm-up per hop. Rotate hop order per fixture to reduce
    # systematic ordering bias; this is a diagnostic, not an isolated benchmark.
    for hop in HOPS:
        analyze_timed(inputs[0][0], hop)
    cases = []
    for index, (samples, metadata) in enumerate(inputs):
        shift = index % len(HOPS)
        for hop in HOPS[shift:] + HOPS[:shift]:
            cases.append(evaluate_samples(samples, {**metadata, "fixture_id": index}, hop))
    return cases


def summarize(cases):
    groups = []
    for hop in HOPS:
        rows = [c for c in cases if c["hop_size"] == hop]
        # Equal weight per fixture: median repeats first, then median across cases.
        costs = [{key: c["timing_summary_ms"][key]["median"] for key in ("features_ms", "tempo_ms", "total_ms")}
                 for c in rows]
        groups.append({"hop_size": hop, "strength_frame_rate": baseline.RATE / hop,
                       "primary": baseline.summarize([c for c in rows if c["group"] == "primary"]),
                       "challenge": baseline.summarize([c for c in rows if c["group"] == "challenge"]),
                       "controls": [{"kind": c["kind"], "estimated_bpm": c["estimated_bpm"],
                                     "score": c["autocorrelation_score"], "strength_peak": c["strength_peak"]}
                                    for c in rows if c["group"] == "control"],
                       "per_case_median_timing_ms": timing_summary(costs)})
    return groups


def save_plots(cases, groups, output):
    figure = Figure(figsize=(13, 5), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    x = np.arange(len(HOPS))
    for offset, key, label in ((-0.18, "primary", "Steady rhythms"), (0.18, "challenge", "Ambiguity cases")):
        bars = left.bar(x + offset, [g[key]["accuracy"] for g in groups], width=0.36, label=label)
        left.bar_label(bars, labels=[f"{g[key]['correct']}/{g[key]['case_count']}" for g in groups], fontsize=9)
    left.set_xticks(x, HOPS)
    left.set(xlabel="Hop size (samples)", ylabel="Nominal-BPM accuracy within 3%", ylim=(0, 1.15))
    left.legend(fontsize=8)
    for key, label in (("features_ms", "STFT + flux"), ("tempo_ms", "Tempo"), ("total_ms", "Total")):
        right.plot(x, [g["per_case_median_timing_ms"][key]["median"] for g in groups], marker="o", label=label)
    right.set_xticks(x, HOPS)
    right.set(xlabel="Hop size (samples)", ylabel="Median per-case runtime (ms)", title="3 repeats/case, medians after warm-up")
    right.legend(fontsize=8)
    figure.suptitle("Same audio and 128-sample frames; unchanged tempo-selection rule")
    figure.savefig(output / "comparison.png", dpi=120)
    figure.clear()
    figure = Figure(figsize=(13, 5), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    for hop in HOPS:
        case = next(c for c in cases if c["hop_size"] == hop and c["group"] == "primary"
                    and c["kind"] == "sharp_burst" and c["nominal_bpm"] == 150
                    and c["gain"] == 1 and c["offset_samples"] == 0)
        bpms = 60 * case["strength_frame_rate"] / np.array(case["lags"])
        left.plot(bpms, case["scores"], marker=".", label=f"Hop {hop}: {case['estimated_bpm']:.2f} BPM")
        right.plot(["Near 150 BPM", "Near 75 BPM"],
                   [case[key]["best_bracketing_bin"]["score"] for key in ("nominal_probe", "half_time_probe")],
                   marker="o", label=f"Hop {hop}")
    for bpm in (75, 150):
        left.axvline(bpm, color="black", linestyle=":", alpha=0.5)
    left.set(xlabel="BPM corresponding to lag", ylabel="Normalized autocorrelation", title="150 BPM sharp burst example")
    left.legend(fontsize=8)
    right.set(ylabel="Best score among floor/ceil of ideal lag", title="Diagnostic probes never influence selection")
    right.legend(fontsize=8)
    figure.savefig(output / "lag-scores.png", dpi=120)
    figure.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = run_comparison()
    groups = summarize(cases)
    hashes = {}
    for c in cases:
        hashes.setdefault(c["fixture_id"], set()).add(c["input_sha256"])
    passed = all(len(h) == 1 for h in hashes.values()) and all(
        np.isfinite(c["scores"]).all() and (c["estimated_bpm"] is None or
        baseline.MIN_BPM <= c["estimated_bpm"] <= baseline.MAX_BPM) for c in cases)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/tempo_hops" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plots(cases, groups, output)
    sources = sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += [Path(__file__), ROOT / "evals/tempo.py"]
    status = baseline.git_output("status", "--porcelain")
    report = {"suite": "tempo_hop_comparison", "suite_version": 1,
              "created_at_utc": now.isoformat(), "label": args.label, "consistency_checks_passed": passed,
              "settings": {"sample_rate": baseline.RATE, "frame_size": baseline.FRAME_SIZE,
                           "hops": HOPS, "repeats": REPEATS, "warmups_per_hop": 1,
                           "min_bpm": baseline.MIN_BPM, "max_bpm": baseline.MAX_BPM,
                           "beat_count": baseline.BEAT_COUNT, "bpms": baseline.BPMS,
                           "relative_bpm_tolerance": baseline.RELATIVE_TOLERANCE,
                           "primary_gains": [0.1, 1.0], "primary_offsets": [0, 11],
                           "alternate_amplitude_ratio": 0.2, "noise_control_seed": 2027},
              "timing_policy": "perf_counter_ns; excludes generation, scoring, hashing, plotting and IO; 3 repeats, rotate hop order per fixture; aggregate median of per-case medians",
              "probe_policy": "floor/ceil of ideal nominal and half-time lags, restricted to actual searched lags; no interpolation or estimator feedback",
              "interpretation": "Changing hop changes spectral flux and frame/lag resolution together; not a pure lag-grid intervention. Timing is diagnostic, not a performance guarantee. Controls remain exploratory.",
              "by_hop": groups, "cases": cases, "python": platform.python_version(),
              "numpy": np.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform(),
              "git_commit": baseline.git_output("rev-parse", "HEAD"),
              "working_tree_dirty": bool(status) if status is not None else None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plots": [str((output / name).relative_to(ROOT)) for name in ("comparison.png", "lag-scores.png")]}
    path = ROOT / "evals/results" / f"tempo-hops-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for group in groups:
        print(f"Hop {group['hop_size']}: primary {group['primary']}; challenge {group['challenge']}")
        print("Median runtime ms:", {k: v["median"] for k, v in group["per_case_median_timing_ms"].items()})
    print(f"Consistency checks: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plots: {output.relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
