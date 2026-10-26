# Audio evaluation history

From the repository root, run:

```sh
PYTHONPATH=src python3 evals/audio_io.py --label "WAV saving baseline"
```

Every run writes a new timestamped JSON report in `evals/results/` and synthetic
input/output WAV files in `outputs/evals/<run-id>/`. Reports can be committed to
keep a history; audio stays local and is ignored by Git. Existing runs are kept.
The script exits with status 1 if any case fails and records the error.

The seven fixed cases cover a mono tone, distinct stereo tones, silence, PCM
limits, empty audio, selection plus half gain, and deliberately clipped gain.
Each case saves, loads, optionally transforms, saves again, and loads the result.
Expected samples are generated independently of the selection/gain functions.
Unit tests additionally check raw PCM bytes, headers, invalid inputs, and writes
over existing files.

To review progress, open two reports and compare cases by `name`:

- `passed` checks exact samples, channel count, and sample rate against expected
  audio. `max_sample_error` and `mismatched_samples` should both be zero.
- Frame count, duration, peak, RMS, and samples at PCM limits describe output.
  A limit-valued sample is not by itself proof of clipping; the clipping case
  deliberately saturates, while the PCM-limit fixture contains those values.
- Input/output SHA-256 hashes help identify changed fixtures or output bytes.
  Compare the same suite version and input hash for meaningful comparisons.
- `elapsed_seconds` covers file I/O and checks for each case. It is a single
  diagnostic measurement, not a stable benchmark; OS caching, machine load,
  and hardware affect it. It is not a pass/fail threshold.

Reports include the Git revision, dirty-tree flag, individual source hashes,
Python version, and platform. Increment `SUITE_VERSION` when fixtures or metric
definitions change. Source hashes identify uncommitted code but do not archive
it; commit the implementation alongside a baseline report for reproducibility.

Listen to `mono_440hz_output.wav`, `stereo_440hz_660hz_output.wav`,
`select_and_half_gain_output.wav`, and `gain_clipping_output.wav` to compare the
examples. PCM-limit and empty fixtures check correctness, not listening quality.
These evaluations establish numerical correctness, not perceived musical quality.

## Direct DFT versus FFT

With the project environment activated, run:

```sh
python evals/dft.py --label "Initial DFT baseline"
```

Seven fixed cases cover silence, DC, a shifted impulse, Nyquist, two tones,
a complex tone of prime length, and a PCM-scale off-bin tone. Reports are saved
as `evals/results/dft-<run-id>.json`; each run preserves earlier results and exits
with status 1 on failure. Compare cases by name, suite version, and input hash.

`max_absolute_error` and `rms_error` measure complex-coefficient differences
from NumPy FFT. A case passes when every bin meets
`abs(actual - reference) <= atol + rtol*abs(reference)`, using atol=1e-8 and
rtol=1e-10. Small nonzero errors are expected from floating-point arithmetic.
Nonfinite output or an incorrect output length fails the case.

Reports retain source hashes, Git state, environment versions, and single-call
DFT/FFT timings. These small timings include conversion overhead and are
diagnostic only; do not treat a one-run timing difference as a performance
regression. Known-answer unit tests separately verify the transform convention.

## Windowing observations

`python examples/windowing.py` generates comparison images and a timestamped
`evals/results/windowing-*.json` report for four cases: 220 Hz and 230 Hz,
each with a rectangular or periodic Hann window. Earlier runs remain intact.
These are observations, not an evaluation with an aggregate pass/fail flag;
`tests/test_windows.py` verifies correctness against known answers and NumPy.

Compare peak amplitude and `max_far_amplitude` across matching case settings.
The latter measures the largest amplitude strictly more than 55 Hz from the
known tone, excluding the wider main lobe. It is not total leakage energy.
The reports retain source hashes, environment versions, and diagnostic single-call
timings. Do not treat small timing changes as performance regressions.
