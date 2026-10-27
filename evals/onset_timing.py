"""Compare frame-start and frame-center onset timing on fixed synthetic cases."""

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

from autodj.analysis import pick_onsets, spectral_flux
from autodj.dsp import stft


ROOT = Path(__file__).resolve().parents[1]
RATE = 2048
SAMPLE_COUNT = 1536
THRESHOLD = 10.0
MIN_INTERVAL = 0.1
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
    errors = None
    if len(start.times) == 1 and len(center.times) == 1:
        # No tolerance gate: retain large errors rather than discarding hard cases.
        errors = {"frame_start": float(start.times[0] - onset_sample / RATE),
                  "frame_center": float(center.times[0] - onset_sample / RATE)}
    return {
        "kind": kind, "frame_size": frame_size, "hop_size": hop_size,
        "offset_within_hop_samples": offset, "known_start_seconds": onset_sample / RATE,
        "threshold": threshold, "max_strength": float(strength.max()),
        "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
        "frame_indices": start.frame_indices.tolist(), "candidate_count": len(start.times),
        "frame_start_times": start.times.tolist(), "frame_center_times": center.times.tolist(),
        "center_shift_seconds": shift,
        "candidate_selection_unchanged": bool(np.array_equal(start.frame_indices, center.frame_indices)),
        "errors_seconds": errors,
    }


def summarize_cases(cases: list[dict]) -> dict:
    paired = [case["errors_seconds"] for case in cases if case["errors_seconds"] is not None]
    methods = {}
    for method in ("frame_start", "frame_center"):
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
            cases.append({"kind": kind, "frame_size": size, "hop_size": hop,
                          "false_candidates": len(candidates.times)})
    return cases


def save_comparison(cases: list[dict], groups: dict, path: Path) -> None:
    figure = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    colors = ("#b35d43", "#197f89")
    labels = ("Frame start", "Frame center")
    for index, (method, label, color) in enumerate(zip(("frame_start", "frame_center"), labels, colors)):
        mae = [groups[kind]["methods"][method]["mean_absolute_error_seconds"] for kind in KINDS]
        left.bar(np.arange(3) + (index - 0.5) * 0.35,
                 [value * 1000 if value is not None else np.nan for value in mae],
                 width=0.35, color=color, label=label)
        for group, kind in enumerate(KINDS):
            errors = [case["errors_seconds"][method] * 1000 for case in cases
                      if case["kind"] == kind and case["errors_seconds"] is not None]
            right.scatter(group * 2 + index + np.linspace(-0.15, 0.15, len(errors)),
                          errors, s=14, color=color, alpha=0.75)
    left.set_xticks(range(3), [f"{kind.replace('_', ' ')}\n"
                              f"{groups[kind]['paired_timing_cases']}/{groups[kind]['case_count']} cases"
                              for kind in KINDS])
    left.set(ylabel="Mean absolute error (ms)", title="Timing error when exactly one peak is selected")
    left.legend()
    right.axhline(0, color="#68737d", linewidth=0.8)
    right.set_xticks(range(6), [f"{kind.replace('_', ' ')}\n{method}"
                              for kind in KINDS for method in ("start", "center")], fontsize=8)
    right.set(ylabel="Signed error (ms): early < 0 < late", title="Individual paired timing errors")
    figure.suptitle("Frame-start vs frame-center onset estimates\n"
                    "Fixed threshold 10 | Missed/multiple detections reported separately")
    figure.savefig(path, dpi=120)
    figure.clear()


def git_output(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
                     and all(case["false_candidates"] == 0 for case in controls))
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onset_timing" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_comparison(cases, groups, output / "comparison.png")
    sources = sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += sorted((ROOT / "src/autodj/analysis").glob("*.py")) + [Path(__file__)]
    status = git_output("status", "--porcelain")
    report = {
        "suite": "onset_timing_comparison", "suite_version": 1,
        "created_at_utc": now.isoformat(), "label": args.label,
        "consistency_and_control_checks_passed": checks_passed,
        "sample_rate": RATE, "sample_count": SAMPLE_COUNT,
        "threshold": THRESHOLD, "min_interval_seconds": MIN_INTERVAL,
        "center_definition": "frame start + frame_size / (2 * sample_rate)",
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
    }
    path = ROOT / "evals/results" / f"onset-timing-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for kind, group in groups.items():
        start = group["methods"]["frame_start"]["mean_absolute_error_seconds"]
        center = group["methods"]["frame_center"]["mean_absolute_error_seconds"]
        print(f"{kind}: {group['paired_timing_cases']}/{group['case_count']} paired cases; "
              f"mean absolute errors (s): start={start}, center={center}; "
              f"center worse in {group['center_worse_cases']} cases")
    print(f"Consistency/control checks: {'PASS' if checks_passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plot: {(output / 'comparison.png').relative_to(ROOT)}")
    return 0 if checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
