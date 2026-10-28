"""Select a threshold on tuning seeds, then evaluate once on held-out noise draws."""

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

if __package__:
    from . import onset_adaptive as adaptive
else:
    import onset_adaptive as adaptive

ROOT = Path(__file__).resolve().parents[1]
TUNING_SEEDS = (101, 211, 307)
HELD_OUT_SEEDS = (401, 503, 607)
GRID = tuple((radius, multiplier) for radius in (8, 12, 16) for multiplier in (1.5, 2.0, 2.5, 3.0))


def objective(group: dict, method: str) -> dict:
    """Micro F1 includes extra predictions from event cases and noise controls."""
    events = group["summary"]["methods"][method]
    controls = group["control_summary"]["methods"][method]
    tp, fn = events["matched_count"], events["missed_count"]
    fp = events["false_count"] + controls["false_count"]
    denominator = 2 * tp + fp + fn
    return {"matched_count": tp, "missed_count": fn, "false_count_including_controls": fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / denominator if denominator else None}


def select_setting(trials: list[dict]) -> dict:
    """Use tuning scores only. Ties: fewer extras, more matches, smaller radius/multiplier."""
    if not trials or any(t["objective"]["f1"] is None for t in trials):
        raise ValueError("Selection requires nonempty trials with defined F1")
    return min(trials, key=lambda t: (-t["objective"]["f1"],
                                     t["objective"]["false_count_including_controls"],
                                     -t["objective"]["matched_count"],
                                     t["radius"], t["multiplier"]))


def run_study() -> dict:
    if set(TUNING_SEEDS) & set(HELD_OUT_SEEDS):
        raise ValueError("Tuning and held-out seeds must be disjoint")
    trials = []
    fixtures = None
    fixed_objective = None
    for radius, multiplier in GRID:
        group = adaptive.evaluate_group(TUNING_SEEDS, radius=radius, multiplier=multiplier)
        if fixtures is None:
            fixtures = [{key: case[key] for key in ("kind", "gain", "noise_std", "seed",
                        "offset_samples", "known_start_seconds", "input_sha256")}
                        for case in group["cases"] + group["controls"]]
            fixed_objective = objective(group, "fixed")
        trials.append({"radius": radius, "multiplier": multiplier,
                       "objective": objective(group, "adaptive"),
                       "summary": group["summary"], "control_summary": group["control_summary"],
                       "by_condition": group["by_condition"], "by_kind": group["by_kind"],
                       "adaptive_case_metrics": [case["metrics"]["adaptive"]
                                                 for case in group["cases"] + group["controls"]]})
    selected = select_setting(trials)
    # Freeze selection before generating or analyzing any held-out fixture.
    selected_config = {"radius": selected["radius"], "multiplier": selected["multiplier"]}
    original = adaptive.evaluate_group(HELD_OUT_SEEDS)
    chosen = adaptive.evaluate_group(HELD_OUT_SEEDS, **selected_config)
    return {"tuning_trials": trials, "tuning_fixture_manifest": fixtures,
            "tuning_fixed_objective": fixed_objective, "selected_config": selected_config,
            "selected_tuning_objective": selected["objective"],
            "held_out_objectives": {"fixed": objective(original, "fixed"),
                                    "original_adaptive": objective(original, "adaptive"),
                                    "selected_adaptive": objective(chosen, "adaptive")},
            "held_out": {"original": original, "selected": chosen}}


def save_plot(study: dict, path: Path) -> None:
    figure = Figure(figsize=(13, 5), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    for radius in (8, 12, 16):
        trials = [t for t in study["tuning_trials"] if t["radius"] == radius]
        left.plot([t["multiplier"] for t in trials], [t["objective"]["f1"] for t in trials],
                  marker="o", label=f"Radius {radius}")
    left.axhline(study["tuning_fixed_objective"]["f1"], linestyle=":", color="#b35d43", label="Fixed 10")
    selected = study["selected_config"]
    left.scatter([selected["multiplier"]], [study["selected_tuning_objective"]["f1"]],
                 marker="*", color="black", s=160, label="Selected", zorder=5)
    left.set(xlabel="Standard-deviation multiplier", ylabel="F1 including noise controls",
             ylim=(0, 1), title="Tuning seeds only: select maximum F1")
    left.legend(fontsize=8)
    names = list(study["held_out_objectives"])
    for i, key in enumerate(("precision", "recall", "f1")):
        values = [study["held_out_objectives"][name][key] for name in names]
        bars = right.bar(np.arange(3) + (i - 1) * 0.25, values, width=0.25, label=key)
        right.bar_label(bars, fmt="%.2f", fontsize=8)
    right.set_xticks(range(3), ["Fixed 10", "Original adaptive", "Selected adaptive"])
    right.set(ylim=(0, 1.12), title="Held-out seeds: no further tuning")
    right.legend(fontsize=8)
    figure.suptitle("Onset threshold selection: separate tuning and evaluation noise draws")
    figure.savefig(path, dpi=120)
    figure.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    study = run_study()
    checks = all(m["matched_count"] + m["missed_count"] == m["event_count"] and
                 m["matched_count"] + m["false_count"] == m["candidate_count"]
                 for group in study["held_out"].values() for case in group["cases"] + group["controls"]
                 for m in case["metrics"].values())
    checks = checks and all(m["matched_count"] + m["missed_count"] == m["event_count"] and
                            m["matched_count"] + m["false_count"] == m["candidate_count"]
                            for trial in study["tuning_trials"] for m in trial["adaptive_case_metrics"])
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/onset_threshold_search" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plot(study, output / "comparison.png")
    sources = sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += [Path(__file__)] + [ROOT / "evals" / name for name in
                                 ("onset_adaptive.py", "onset_sensitivity.py", "onset_timing.py")]
    status = adaptive.baseline.git_output("status", "--porcelain")
    report = {"suite": "onset_threshold_search", "suite_version": 1,
              "created_at_utc": now.isoformat(), "label": args.label,
              "accounting_checks_passed": checks,
              "settings": {"tuning_seeds": TUNING_SEEDS, "held_out_seeds": HELD_OUT_SEEDS,
                           "grid": GRID, "floor": adaptive.FLOOR,
                           "fixed_threshold": adaptive.baseline.THRESHOLD,
                           "original_adaptive": {"radius": adaptive.RADIUS, "multiplier": adaptive.MULTIPLIER},
                           "sample_rate": adaptive.baseline.RATE, "sample_count": adaptive.baseline.SAMPLE_COUNT,
                           "frame_size": adaptive.baseline.FRAME_SIZE, "hop_size": adaptive.baseline.HOP_SIZE,
                           "gains": adaptive.baseline.GAINS, "noise_stds": adaptive.baseline.NOISE_STDS,
                           "offsets_samples": adaptive.baseline.OFFSETS,
                           "noise_generator": "NumPy PCG64 standard_normal; paired noise across conditions",
                           "min_interval_seconds": adaptive.baseline.MIN_INTERVAL,
                           "match_tolerance_seconds": adaptive.baseline.TOLERANCE,
                           "timing": "frame center, no energy refinement"},
              "selection_rule": "Maximum tuning micro F1 including noise controls; ties: fewer false, more matched, smaller radius, smaller multiplier. No held-out feedback.",
              **study, "python": platform.python_version(), "numpy": np.__version__,
              "matplotlib": matplotlib.__version__, "platform": platform.platform(),
              "git_commit": adaptive.baseline.git_output("rev-parse", "HEAD"),
              "working_tree_dirty": bool(status) if status is not None else None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plot": str((output / "comparison.png").relative_to(ROOT))}
    path = ROOT / "evals/results" / f"onset-threshold-search-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(f"Selected: {study['selected_config']}")
    for name, scores in study["held_out_objectives"].items():
        print(f"Held out / {name}: {scores}")
    print(f"Accounting checks: {'PASS' if checks else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plot: {(output / 'comparison.png').relative_to(ROOT)}")
    return 0 if checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
