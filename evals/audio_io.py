"""Repeatable synthetic audio evaluation; run from the repository root."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from autodj.audio import WavAudio, apply_gain, load_wav, save_wav, select_time_range


ROOT = Path(__file__).resolve().parents[1]
SUITE_VERSION = 1


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cases():
    """Fixed inputs and expected outputs; no random or commercial audio."""
    rate = 8000
    mono = WavAudio(rate, 1, tuple(
        (round(12000 * math.sin(2 * math.pi * 440 * n / rate)),)
        for n in range(rate)
    ))
    stereo = WavAudio(rate, 2, tuple(
        (frame[0], round(6000 * math.sin(2 * math.pi * 660 * n / rate)))
        for n, frame in enumerate(mono.samples)
    ))
    for name, audio in (
        ("mono_440hz", mono), ("stereo_440hz_660hz", stereo),
        ("silence", WavAudio(44100, 1, ((0,),) * 4410)),
        ("pcm_limits", WavAudio(48000, 2, ((-32768, 32767), (-1, 1), (0, 0)))),
        ("empty", WavAudio(8000, 1, ())),
    ):
        yield name, audio, audio, None

    # Independent expected frames: 0.25–0.75 seconds at 8 kHz, half gain.
    expected = WavAudio(rate, 2, tuple(
        tuple(round(sample / 2) for sample in frame)
        for frame in stereo.samples[2000:6000]
    ))
    yield "select_and_half_gain", stereo, expected, (0.25, 0.75, 0.5)
    # An audible tone that deliberately clips at both PCM limits.
    expected = WavAudio(rate, 1, tuple(
        (max(-32768, min(32767, frame[0] * 4)),) for frame in mono.samples
    ))
    yield "gain_clipping", mono, expected, (0.0, 1.0, 4.0)


def evaluate(name, source, expected, operation, output_dir):
    result = {"name": name, "operation": operation, "passed": False}
    start = perf_counter()
    try:
        input_path = output_dir / f"{name}_input.wav"
        output_path = output_dir / f"{name}_output.wav"
        save_wav(source, input_path)
        audio = load_wav(input_path)
        if operation is not None:
            begin, end, gain = operation
            audio = apply_gain(select_time_range(audio, begin, end), gain)
        save_wav(audio, output_path)
        actual = load_wav(output_path)
        shape_matches = (
            actual.frame_count == expected.frame_count
            and actual.channels == expected.channels
        )
        errors = [
            abs(a - b)
            for actual_frame, expected_frame in zip(actual.samples, expected.samples)
            for a, b in zip(actual_frame, expected_frame)
        ] if shape_matches else None
        values = [sample for frame in actual.samples for sample in frame]
        result.update(
            passed=actual == expected,
            sample_rate=actual.sample_rate,
            channels=actual.channels,
            frame_count=actual.frame_count,
            duration_seconds=actual.duration,
            expected_frame_count=expected.frame_count,
            max_sample_error=max(errors, default=0) if errors is not None else None,
            mismatched_samples=sum(e != 0 for e in errors) if errors is not None else None,
            peak_amplitude=max(map(abs, values), default=0),
            rms_amplitude=math.sqrt(sum(v * v for v in values) / len(values)) if values else 0,
            samples_at_pcm_limits=sum(v in (-32768, 32767) for v in values),
            output_bytes=output_path.stat().st_size,
            input_sha256=sha256(input_path.read_bytes()),
            output_sha256=sha256(output_path.read_bytes()),
            input_wav=str(input_path.relative_to(ROOT)),
            output_wav=str(output_path.relative_to(ROOT)),
        )
    except Exception as exc:
        # Retain failed cases in the report and continue evaluating the others.
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(perf_counter() - start, 6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="", help="Optional note stored with this run")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = ROOT / "outputs" / "evals" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    results = [evaluate(*case, output_dir) for case in cases()]
    source_files = sorted((ROOT / "src" / "autodj").rglob("*.py")) + [Path(__file__)]
    status = git_output("status", "--porcelain", "--untracked-files=normal")
    report = {
        "suite": "audio_io", "suite_version": SUITE_VERSION,
        "run_id": run_id, "created_at_utc": now.isoformat(), "label": args.label,
        "git_commit": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in source_files
        },
        "python": sys.version, "platform": platform.platform(),
        "passed": all(case["passed"] for case in results), "cases": results,
    }
    report_dir = ROOT / "evals" / "results"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}.json"
    with report_path.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    for case in results:
        print(f"{'PASS' if case['passed'] else 'FAIL'} {case['name']} "
              f"({case['elapsed_seconds']:.6f}s)")
    print(f"Report: {report_path.relative_to(ROOT)}")
    print(f"Audio: {output_dir.relative_to(ROOT)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
