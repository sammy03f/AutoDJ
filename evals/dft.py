"""Save a repeatable numerical comparison of the direct DFT and NumPy FFT."""

import argparse
import cmath
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from autodj.dsp import dft


ROOT = Path(__file__).resolve().parents[1]
SUITE_VERSION = 1
ATOL, RTOL = 1e-8, 1e-10


def cases():
    yield "silence", [0] * 32
    yield "constant", [2] * 32
    yield "shifted_impulse", [0, 1] + [0] * 30
    yield "nyquist", [1, -1] * 16
    yield "two_tones", [
        math.sin(2 * math.pi * 8 * n / 64) + 0.5 * math.sin(2 * math.pi * 16 * n / 64)
        for n in range(64)
    ]
    yield "complex_prime_length", [cmath.exp(2j * math.pi * 3 * n / 31) for n in range(31)]
    yield "pcm_off_bin", [round(30000 * math.sin(2 * math.pi * 3.5 * n / 64)) for n in range(64)]


def evaluate(name, samples):
    encoded = json.dumps([[complex(x).real, complex(x).imag] for x in samples]).encode()
    result = {"name": name, "sample_count": len(samples),
              "input_sha256": hashlib.sha256(encoded).hexdigest(), "passed": False}
    try:
        start = perf_counter()
        actual = np.asarray(dft(samples))
        result["dft_seconds"] = perf_counter() - start
        start = perf_counter()
        expected = np.fft.fft(samples)
        result["fft_seconds"] = perf_counter() - start
        if actual.shape != expected.shape:
            raise ValueError("DFT output length differs from FFT")
        if not np.isfinite(actual).all():
            raise ValueError("DFT produced nonfinite coefficients")
        error = np.abs(actual - expected)
        result.update(
            passed=bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
            max_absolute_error=float(error.max()),
            rms_error=float(np.sqrt(np.mean(error ** 2))),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


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
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    results = [evaluate(name, samples) for name, samples in cases()]
    status = git_output("status", "--porcelain")
    sources = sorted((ROOT / "src/autodj/dsp").glob("*.py")) + [Path(__file__)]
    report = {
        "suite": "dft", "suite_version": SUITE_VERSION, "run_id": run_id,
        "created_at_utc": now.isoformat(), "label": args.label,
        "git_commit": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "atol": ATOL, "rtol": RTOL,
        "passed": all(case["passed"] for case in results), "cases": results,
    }
    path = ROOT / "evals/results" / f"dft-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    for case in results:
        print(f"{'PASS' if case['passed'] else 'FAIL'} {case['name']}")
    print(f"Report: {path.relative_to(ROOT)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
