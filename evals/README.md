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
