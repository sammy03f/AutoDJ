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

## Onset timing comparison

Run `python evals/onset_timing.py --label "Timing baseline"` from the activated
project environment. It saves a timestamped report under
`evals/results/onset-timing-*.json` and comparison/energy-example PNGs under `outputs/onset_timing/`.
Reports retain source fingerprints, Git state, versions, settings, and per-case
input hashes. Earlier reports and images are preserved.

The fixed suite has 108 positive cases: 3 shapes × 3 frame lengths × 3 hop
ratios × 4 within-hop offsets. Signals are 1536 samples at 2048 Hz; frames are
64/128/256 samples, hops are L/8, L/4, L/2, and offsets are 0, H/4, H/2, H-1.
Shapes are a click, a sharp decaying cosine, and a cosine with a slower attack.
The first nonzero sample is ground truth. Constants and formulas live in the
script and are covered by fixture tests. Threshold 10 and minimum interval
0.1 s are held fixed. An additional 18 silence/steady-tone controls are checked.

Frame center is start + L/(2*Fs). Peak selection is run with each time convention
to check that indices do not change. Errors are candidate minus known onset.
Only cases with exactly one candidate receive paired timing errors, without
a matching tolerance gate. Missing/multiple cases have null errors, never zero.
Reports include their counts, signed/absolute/max errors, cases that improve or
worsen, and within-tolerance counts with all positive cases in the denominator.
The multiple-candidate category is excluded from timing comparisons rather than
choosing whichever candidate is closest to ground truth.

Initial findings: 100/108 paired cases, 8 misses, no multiple detections, and no
false candidates in the controls. Center labeling reduces paired mean absolute
error from 40.60 to 11.25 ms, but worsens 12 slow-attack cases. These findings
support an experimental alternative, not a general accuracy guarantee or a
change to the default timestamps. The sweep does not cover every phase, sample
rate, gain, noise condition, or polyphonic signal.

Suite version 2 preserves the original signals and detection settings and adds
energy refinement. It uses trailing mean-square energy over 8 samples, a 10%
threshold above the local energy minimum, and a search radius of one STFT frame
duration around the center timestamp. These constants are fixed across cases;
ground-truth timestamps are used only to score results. The first upward crossing
before the earliest local maximum is selected; no crossing retains the center.
The extra energy-example plot shows this rule on a slow attack.

Refined paired mean absolute error is 3.14 ms (clicks 0, sharp bursts 0, slow
attacks 11.23 ms). Relative to centers, 84 cases improve, 7 worsen, and 9 tie;
8 events remain missed. All 100 detected cases are within 20 ms, scored as
100/108 across all events. Exact click/burst timing relies on these fixtures'
silent backgrounds and abrupt starts. Noise, overlapping events, and different
carrier phases remain untested by this suite. Energy refinement can follow an
unrelated rise, and overlapping searches can produce duplicate/reordered times.

Reports from version 1 retain only the two original methods. Compare those
shared fields across versions; compare refined results against version 2 runs
with matching settings and fixture hashes.

Exit status checks selection invariance, refinement count, and negative controls only. A passing
run does **not** mean every event was detected or refinement always improved
timing. Compare matching suite/settings and inspect failures alongside error
averages. Tests in `tests/test_onset_timing_eval.py` verify shift conventions,
fixture starts, null errors for misses, and summary denominators.


## Volume and noise sensitivity

Run `python evals/onset_sensitivity.py --label "Fixed threshold baseline"`.
It writes `evals/results/onset-sensitivity-<run-id>.json` and a comparison PNG
under `outputs/onset_sensitivity/<run-id>/`. Earlier runs are preserved. JSON
includes input/source hashes, Git state, library versions, full settings,
per-case timestamps/metrics, pooled summaries, and gain/noise and shape groups.

Suite version 1 has 576 single-event cases:

- Three existing shapes: click, sharp burst, slow attack.
- Event gains 0.1, 0.25, 0.5, 1.0; additive Gaussian noise standard deviations
  0, 0.01, 0.05, 0.2. Gain scales the event only, not the noise.
- NumPy PCG64 seeds 7, 19, 43, reused across shapes/gains/offsets for paired
  comparisons. At zero noise the seed repetitions are intentionally identical.
- Event offsets 0, 8, 16, 31 samples after sample 512; 1536 samples at 2048 Hz.
- Fixed STFT frame 128/hop 32, flux threshold 10, minimum spacing 0.1 s.
- Refinement uses 8-sample trailing energy, local fraction 0.1, and a search
  radius of 128/2048 s around each frame center.

There are also 12 noise-only controls (four noise levels × three seeds), without
a declared event. Noise standard deviation is not an SNR; realized noise RMS is
recorded. Signals are floating-point arrays with no clipping or normalization.
The known event time is the clean signal's first nonzero sample, even when noise
is present before that time. The experiment does not change the detector.

Each method independently matches at most one candidate: the nearest to the
single event within an inclusive 50 ms tolerance (1e-12 s roundoff allowance).
Equal-distance ties choose the first candidate. Every remaining candidate is a
false detection. If no candidate matches, the event is missed; an out-of-range
candidate counts as false as well. All candidates in controls are false.

Precision = matches / candidate count; recall = matches / event count. Summaries
pool counts, rather than average per-case percentages. Undefined ratios are
null, including control recall and precision with no candidates. Timing-error
means use matched candidates only, with their count reported; methods can match
different subsets, so their error averages are not a paired timing comparison.
False detections per second use the complete generated signal duration, including
padding; boundary suppression and discarded STFT tails remain part of the pipeline.

Initial event-case results:

| Timestamp method | Matches / 576 | Misses | False detections |
| --- | --- | --- | --- |
| Frame start | 401 | 175 | 590 |
| Frame center | 403 | 173 | 588 |
| Energy refined | 342 | 234 | 649 |

At gain 0.1 in silence, all 36 event cases are missed. At gains 0.5 and 1.0 in
silence, all 36 cases per gain match. At noise standard deviation 0.2 the controls
produce 15 false detections in 2.25 seconds, or 6.67/s; lower tested noise levels
produce none in these controls. Refinement can move candidates to unrelated
energy rises and makes this noisy sweep worse despite its clean-signal benefit.

High recall under strong noise can include chance temporal matches and does not
prove the detector recognized the event. Read precision, false counts, and controls
together. Seed reuse means these are correlated fixtures, not 576 independent
trials or a confidence estimate. Only one sample rate, STFT setting, carrier
phase, and stationary white-noise model are covered. This is not real-music
validation, and future tuning needs fresh seeds/cases as well as this baseline.

Exit status checks candidate-count preservation and scoring accounting only;
misses and false detections do not fail the evaluation. Unit tests verify noise
repeatability, gain independence, tolerance boundaries, duplicate accounting,
null metrics, pooled counts, clean detection, and noise controls.


## Fixed versus adaptive thresholds

Run `python evals/onset_adaptive.py --label "Local mean plus 3 std baseline"`.
Reports are `evals/results/onset-adaptive-<run-id>.json`; comparison and threshold
plots live under `outputs/onset_adaptive/<run-id>/`. Full per-case strength,
threshold, timestamps, input/source hashes, settings, Git state and environment
versions are retained. The plots include a quiet success, a noisy success, and
a clean slow-attack failure; these examples are not aggregate accuracy evidence.

Suite version 1 reuses the sensitivity suite's shapes, gains, noise levels,
offsets, sample rate, STFT settings, spacing, and 50 ms matching rule. Each arm
uses frame-center timestamps with no energy refinement. Only the threshold
changes: fixed 10 versus `max(0.1, mean + 3 * population_std)` in a centered
radius-8-frame neighborhood including the current frame, clipped at boundaries.
Parameters were declared before running either seed group; no parameter sweep
or tuning is performed. The curve is offline and includes future frames.

Baseline seeds are 7/19/43; fresh seeds are 101/211/307. Each group contains
576 event cases and 12 noise-only controls. Cases reuse noise across conditions,
and zero-noise fixtures are identical across seed groups. Fresh seeds test new
noise draws, not new shapes, phases, rates, or independent clean examples. They
are now observed evaluation data; future tuning needs another untouched group.

| Seeds | Threshold | Matches / 576 | Misses | Extra candidates | Noise-only candidates |
| --- | --- | --- | --- | --- | --- |
| Baseline | Fixed | 403 | 173 | 588 | 15 |
| Baseline | Adaptive | 267 | 309 | 0 | 0 |
| Fresh | Fixed | 360 | 216 | 603 | 15 |
| Fresh | Adaptive | 262 | 314 | 0 | 0 |

At gain 0.1 in silence, adaptive selection recovers 18/36 cases versus 0/36
for fixed selection. At gains 0.5 and 1.0 in silence it matches only 18/36 versus
36/36 for fixed selection. Including broad events in their own neighborhood
can raise the threshold enough to reject them. Eliminating false detections is
therefore accompanied by substantial lost recall. The adaptive option is not
promoted to a default, and zero false detections here does not establish noise
robustness on music or other noise types.

Scoring is independent per detector: nearest candidate within tolerance, all
extras false, and unmatched truth missed. Conditional timing means may use
different matched populations. Noise can create chance matches for the fixed
arm, so its larger match count alone is not proof of better event recognition.
Exit status checks accounting, not improved accuracy. Tests verify the local
statistics, edge clipping, strict per-frame thresholds, plateau semantics,
scalar compatibility, invalid inputs, seed separation, and fixed-arm parity
with the original sensitivity evaluation.


## Selecting an adaptive threshold

Run `python evals/onset_threshold_search.py --label "Threshold selection baseline"`.
The JSON report is saved as `evals/results/onset-threshold-search-<run-id>.json`,
with a plot under `outputs/onset_threshold_search/<run-id>/`. Suite version 1
uses radii 8/12/16 and multipliers 1.5/2/2.5/3 (12 combinations), always floor
0.1. Other signal, detection and scoring settings match the adaptive evaluation:
576 isolated event cases and 12 noise-only controls per seed group, frame-center
timestamps, no energy refinement, and 50 ms matching tolerance.

Tuning uses previously observed seeds 101/211/307. For each setting, pool matches
TP, misses FN, and false candidates FP from both event cases and controls, then
compute `F1 = 2*TP / (2*TP + FP + FN)`. This balances precision and recall under
the suite's mixture of event/noise-only cases; changing that mixture could
change the preferred setting. Ties prefer fewer false candidates, more matches,
then smaller radius and multiplier. Zero predictions with events yield F1 zero;
a wholly empty scoring population has undefined F1 and cannot be selected.

The winning configuration is frozen before generating/analyzing held-out seeds
401/503/607. Only the fixed, original adaptive, and selected adaptive settings
are evaluated on those draws. Held-out scores do not trigger reselection. All
12 tuning summaries and ordered per-case adaptive metrics are retained, with
an input-hash manifest (event cases followed by controls). The held-out sections
retain full case details, curves and timestamps. Source hashes and environment
versions support later comparisons.

The selected radius is 12 frames, multiplier 2.5, floor 0.1. Tuning F1 is 0.840.
The original adaptive baseline is radius 8 / multiplier 3 / floor 0.1.

| Held-out method | Matches / 576 | Misses | False including controls | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed 10 | 402 | 174 | 649 | 0.382 | 0.698 | 0.494 |
| Original adaptive | 263 | 313 | 0 | 1.000 | 0.457 | 0.627 |
| Selected adaptive | 425 | 151 | 0 | 1.000 | 0.738 | 0.849 |

The fixed arm's 649 false candidates comprise 633 on event cases and 16 on
noise-only controls. The selected setting matches all clean events at all four
gains, but still misses quiet events in stronger noise. There is no default
change: this is a promising configuration for these fixtures, not proof of
robustness on songs, overlapping events, or changing background noise.

Seeds are disjoint, but the signal families, gains, phases, and event positions
are shared. Zero-noise cases repeat exactly between groups. This checks new
noise realizations, not independent musical diversity; seeds are reused across
conditions, so cases are correlated. The held-out seeds are now observed and
must not become a feedback loop for further tuning without a new evaluation set.
Tests cover objective accounting, tie-breaking, seed overlap rejection, unchanged
baseline behavior, and the order of tuning/selection/held-out evaluation. Exit
status checks accounting, not an accuracy target.
