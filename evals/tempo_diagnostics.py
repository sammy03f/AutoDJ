"""Inspect residual tempo acceptance across gain and time windows; no new gate."""

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

from autodj.analysis import estimate_tempo, spectral_flux
from autodj.dsp import stft

if __package__:
    from . import tempo_evidence as evidence
else:
    import tempo_evidence as evidence

baseline = evidence.baseline
ROOT = Path(__file__).resolve().parents[1]
SCALES = (1., 0.01, 1e-4, 1e-6)
AGREEMENT_TOLERANCE = 0.03


def relative_peak_flux(strength, coefficients):
    """Diagnostic peak flux / mean per-frame sum of raw FFT magnitudes.

    Both numerator and denominator scale with gain. This is not a probability,
    has no acceptance threshold, and depends on window/hop and signal occupancy.
    Silence or no complete frames yields None, not a fabricated zero ratio.
    """
    if not len(strength):
        return None
    denominator = float(np.abs(coefficients).sum(axis=1).mean())
    return float(np.max(strength) / denominator) if denominator > 0 else None


def estimate_fields(strength, *, gated=False):
    result = estimate_tempo(strength, baseline.RATE / evidence.HOP_SIZE,
                            min_bpm=baseline.MIN_BPM, max_bpm=baseline.MAX_BPM,
                            min_peak_strength=evidence.PEAK_FLOOR if gated else 0,
                            min_correlation=evidence.CORRELATION_FLOOR if gated else 0)
    return {"bpm": result.bpm, "lag_frames": result.lag_frames, "score": result.score,
            "rejection_reason": result.rejection_reason,
            "frame_count": len(strength), "searched_lags": result.lags.tolist(),
            "scores": result.scores.tolist(),
            "period_lengths_at_estimate": len(strength) / result.lag_frames if result.lag_frames else None}


def window_agreement(full, windows):
    """None means at least one estimate is absent; no octave equivalence is used."""
    if full["bpm"] is None or any(w["bpm"] is None for w in windows):
        return None
    return all(abs(w["bpm"] - full["bpm"]) / full["bpm"] <= AGREEMENT_TOLERANCE + 1e-12 for w in windows)


def diagnose(samples, metadata):
    spectrum = stft(samples, baseline.RATE, frame_size=baseline.FRAME_SIZE, hop_size=evidence.HOP_SIZE)
    strength = spectral_flux(spectrum)
    full = estimate_fields(strength)
    midpoint = len(strength) // 2
    # Slice the already-computed curve: no newly fabricated flux at the split.
    windows = [estimate_fields(strength[:midpoint]), estimate_fields(strength[midpoint:])]
    return {**metadata, "duration_seconds": len(samples) / baseline.RATE,
            "input_sha256": hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
            "strength_peak": float(strength.max()) if strength.size else 0.,
            "mean_spectral_magnitude": float(np.abs(spectrum.coefficients).sum(axis=1).mean()) if strength.size else 0.,
            "relative_peak_flux": relative_peak_flux(strength, spectrum.coefficients),
            "full": full, "gated": estimate_fields(strength, gated=True),
            "half_windows": windows, "half_split_frame": midpoint,
            "half_window_agreement": window_agreement(full, windows)}


def fixtures():
    rows = [(samples, {"group": "control", **metadata}) for samples, metadata in evidence.control_signals()]
    for kind in ("click", "sharp_burst"):
        for bpm in (90, 120, 180):
            samples, _ = baseline.make_rhythm(kind, bpm)
            rows.append((samples, {"group": "rhythm", "kind": kind, "nominal_bpm": bpm}))
    return rows


def summarize(rows):
    accepted = [r for r in rows if r["gated"]["bpm"] is not None]
    return {"case_count": len(rows), "baseline_accepted": sum(r["full"]["bpm"] is not None for r in rows),
            "gated_accepted": len(accepted), "gated_abstained": len(rows) - len(accepted),
            "accepted_agreement": sum(r["half_window_agreement"] is True for r in accepted),
            "accepted_disagreement": sum(r["half_window_agreement"] is False for r in accepted),
            "accepted_incomplete_windows": sum(r["half_window_agreement"] is None for r in accepted)}


def save_plot(cases, path):
    figure = Figure(figsize=(15, 5), layout="constrained")
    FigureCanvasAgg(figure)
    raw, relative, windows = figure.subplots(1, 3)
    representatives = [next(c["fixture_id"] for c in cases if c["kind"] == "steady_tone" and
                            c.get("frequency_hz") == 137 and c.get("gain") == 1 and c["duration_seconds"] == 8),
                       next(c["fixture_id"] for c in cases if c["group"] == "rhythm" and
                            c["kind"] == "sharp_burst" and c["nominal_bpm"] == 120)]
    for fixture, label in zip(representatives, ("Steady 137 Hz tone", "120 BPM burst rhythm")):
        rows = [c for c in cases if c["fixture_id"] == fixture]
        raw.plot([c["scale"] for c in rows], [c["strength_peak"] for c in rows], marker="o", label=label)
        relative.plot([c["scale"] for c in rows], [c["relative_peak_flux"] for c in rows], marker="o", label=label)
    raw.axhline(evidence.PEAK_FLOOR, color="black", linestyle=":", label="Existing raw floor .001")
    raw.set(xscale="log", yscale="log", xlabel="Gain multiplier", ylabel="Raw peak flux", title="Raw strength depends on gain")
    relative.set(xscale="log", yscale="log", xlabel="Gain multiplier", ylabel="Peak flux / mean spectral magnitude",
                 title="Relative flux is diagnostic only")
    raw.legend(fontsize=8)
    relative.legend(fontsize=8)
    irregular = next(c["fixture_id"] for c in cases if c["kind"] == "irregular_bursts" and
                     c["duration_seconds"] == 2 and c["seed"] == 2027)
    for fixture, label in zip(representatives + [irregular], ("Steady tone", "Rhythm", "2 s irregular bursts")):
        c = next(c for c in cases if c["fixture_id"] == fixture and c["scale"] == 1)
        values = [r["bpm"] if r["bpm"] is not None else np.nan for r in [c["full"], *c["half_windows"]]]
        windows.plot(range(3), values, marker="o", label=label)
    windows.set_xticks(range(3), ["Full", "First half", "Second half"])
    windows.set(ylabel="Ungated tempo candidate (BPM)", title="Window stability is not sufficient evidence")
    windows.legend(fontsize=8)
    figure.suptitle("Residual tempo diagnostics | frozen estimator and evidence checks")
    figure.savefig(path, dpi=120)
    figure.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = [diagnose(samples * scale, {**metadata, "fixture_id": i, "scale": scale})
             for i, (samples, metadata) in enumerate(fixtures()) for scale in SCALES]
    # Acceptance is unchanged by this diagnostic: each accepted gated estimate
    # must equal the ungated candidate. Finite ratios or explicit None are retained.
    passed = all((c["gated"]["bpm"] is None or c["gated"]["bpm"] == c["full"]["bpm"])
                 and (c["relative_peak_flux"] is None or np.isfinite(c["relative_peak_flux"])) for c in cases)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "outputs/tempo_diagnostics" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_plot(cases, output / "comparison.png")
    sources = sorted((ROOT / "src/autodj/analysis").glob("*.py")) + sorted((ROOT / "src/autodj/dsp").glob("*.py"))
    sources += [Path(__file__), ROOT / "evals/tempo.py", ROOT / "evals/tempo_evidence.py"]
    status = baseline.git_output("status", "--porcelain")
    report = {"suite": "tempo_residual_diagnostics", "suite_version": 1,
              "created_at_utc": now.isoformat(), "label": args.label, "consistency_checks_passed": passed,
              "settings": {"sample_rate": baseline.RATE, "frame_size": baseline.FRAME_SIZE,
                           "hop_size": evidence.HOP_SIZE, "min_bpm": baseline.MIN_BPM, "max_bpm": baseline.MAX_BPM,
                           "min_peak_strength": evidence.PEAK_FLOOR, "min_correlation": evidence.CORRELATION_FLOOR,
                           "gain_multipliers": SCALES, "window_agreement_tolerance": AGREEMENT_TOLERANCE,
                           "control_seeds": evidence.SEEDS, "control_durations_seconds": evidence.DURATIONS},
              "method": "Scale identical waveforms; compare raw peak with peak/mean summed FFT magnitude. Split computed strength at floor(N/2); use ungated estimates in each half, no reset of first flux value. No new acceptance rule or tuning.",
              "limitations": "Shared/observed fixtures, not held-out validation. Relative metric depends on FFT/hop and signal occupancy. Short halves can lack slow-tempo support. Adjacent STFT windows overlap in samples. Stable artifacts can agree across windows; instability is not proof of no rhythm.",
              "summary": summarize(cases),
              "by_scale": [{"scale": scale,
                            "rhythms": summarize([c for c in cases if c["scale"] == scale and c["group"] == "rhythm"]),
                            "controls": summarize([c for c in cases if c["scale"] == scale and c["group"] == "control"])} for scale in SCALES],
              "by_kind": {kind: summarize([c for c in cases if c["kind"] == kind]) for kind in sorted({c["kind"] for c in cases})},
              "cases": cases, "python": platform.python_version(), "numpy": np.__version__,
              "matplotlib": matplotlib.__version__, "platform": platform.platform(),
              "git_commit": baseline.git_output("rev-parse", "HEAD"), "working_tree_dirty": bool(status) if status is not None else None,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "plot": str((output / "comparison.png").relative_to(ROOT))}
    path = ROOT / "evals/results" / f"tempo-diagnostics-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(report, destination, indent=2, allow_nan=False)
        destination.write("\n")
    for row in report["by_scale"]:
        print(row)
    print(f"Consistency checks: {'PASS' if passed else 'FAIL'}")
    print(f"Report: {path.relative_to(ROOT)}")
    print(f"Plot: {(output / 'comparison.png').relative_to(ROOT)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
