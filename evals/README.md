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

## STFT tone-switch check

`python examples/stft.py` retains a timestamped `evals/results/stft-*.json`
report and the input/output arrays in `outputs/stft/<run-id>/tone_switch.npz`.
The fixture switches from 64 Hz to 192 Hz at 0.25 seconds. It uses 64-sample
Hann frames with a 32-sample hop at 1024 Hz (15 frames, 33 bins).

A run passes when complex coefficients match an independently framed direct-DFT
reference within atol=rtol=1e-10, start timestamps match, and the wholly contained
frames identify the right tone. The frame spanning the switch is included in
the DFT comparison but excluded from the pure-tone peak checks. The script exits
with status 1 on a failed comparison. Reports record source fingerprints and
environment versions; compare matching suite versions. Full boundary and
amplitude checks are in `tests/test_stft.py`.

The example now also saves a spectrogram PNG beside the arrays. Its report
records the path, reference amplitude, color limits, column width, Matplotlib
version, and renderer source hash. The numerical suite and its pass/fail criteria
remain unchanged. `tests/test_spectrogram.py` separately checks dB scaling,
DC/Nyquist handling, array orientation, cell coordinates, silence, and PNG output.
Compare numeric results rather than PNG byte hashes across Matplotlib versions.

## Spectral-flux bursts

`python examples/onsets.py` retains a plot and NumPy arrays under
`outputs/onsets/<run-id>/` and an `evals/results/onsets-<run-id>.json` report.
Three known damped-tone bursts start at 0.25, 0.75, and 1.25 seconds. Reports
record fixture settings, source hashes, versions, and maximum strength in each
known region. Those regions include current/previous frames overlapping a burst;
they are not found by an onset detector.

The curve checks pass if it matches a scalar reference (atol=rtol=1e-10), stays
nonnegative, responds within each known burst region, and is zero within 1e-12
where both compared frames are silent. The scalar reference tests vectorization,
not the algorithm independently; known-answer tests in `tests/test_onsets.py`
verify the formula and exact impulse responses.

Suite version 2 adds peak picking with threshold 10 and minimum interval 0.1 s.
The plot marks selected candidates, and the arrays retain their indices/times.
Chronological candidates are paired one-to-one with the nearest unmatched known
event within 31.25 ms (two hops). Reports include extra/missed events, precision,
recall, and signed/absolute timing error. Unmatched candidates or missed events
make the run fail, even if the flux calculation passed. Failure exits with status 1.
The current fixture has three matches and no extras, with every estimate 31.25 ms
early. This is a frame-start baseline; no correction is applied before scoring.
The tolerance is part of the metric, not a claim of exact timing or performance
on real music. Empty matches produce null timing-error summaries.

Compare matching fixture/settings/suite versions. Version 1 reports remain
valid records of strength-only evaluation; they contain no peak metrics. Peak
rule tests (silence, boundaries, plateaus, thresholds, ties, and close events)
are in `tests/test_peak_picking.py`. BPM evaluation is still future work.
