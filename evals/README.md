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


## Repeated events at known tempos

Run `python evals/onset_sequences.py --label "Repeated events baseline"`.
Reports are saved as `evals/results/onset-sequences-<run-id>.json`, with comparison
and example sequence PNGs under `outputs/onset_sequences/<run-id>/`. Source/input
hashes, versions, Git state, detector settings, per-hit matching and per-condition
summaries are retained. There is no retuning and BPM is not estimated.

Suite version 1 has 288 eight-hit event tracks (2304 known hits):

- Clicks and sharp 128 Hz decaying bursts, using the existing burst formula.
- 60/120/180 BPM, with one or four hits per beat.
- Steady amplitudes or alternating 1/0.35 amplitudes, multiplied by gain 0.1 or 1.
- Additive Gaussian noise std 0/0.05/0.2, PCG64 seeds 701/809.
- Fs=2048, first event at sample 512; each absolute event position is rounded
  independently, avoiding accumulated interval-rounding drift. Half a second
  of audio follows the final event start. No normalization or clipping.
- Frozen STFT 128/hop 32, fixed threshold 10, original adaptive radius 8 / 3 std,
  selected adaptive radius 12 / 2.5 std, floor 0.1, minimum spacing 0.1 s.
  All methods use frame-center timestamps without energy refinement.

Thirty-six noise-only controls cover the six track durations, three noise
levels and two seeds. They have no event truth; every candidate is false.
Noise is paired across compatible conditions. Zero-noise seed repetitions and
some equal-duration controls are identical; these are correlated fixtures,
not independent trials. Absolute event positions vary relative to the STFT hop
at 180 BPM. Stationary white noise, two event shapes and fixed carrier phase
remain much narrower than real music.

Matching uses a chronological one-to-one dynamic program. It first maximizes
the number of pairs within 50 ms (inclusive with 1e-12 s roundoff allowance),
then minimizes summed absolute timing error. Exact ties prefer matching, skipping
a candidate, then skipping a truth. It handles overlapping tolerance windows
without assigning the same candidate or event twice. This matters at 83.3 ms hit
spacing. Every unmatched truth is a miss and every unmatched candidate is false.
Precision/recall/F1 pool hit counts. Timing means include matches only, and each
method may match different hits. A brute-force oracle checks short matching
problems independently of the implementation.

The 48 tracks at 180 BPM with four hits/beat contain 384 hits at about 83.3 ms
spacing, less than the detector's 100 ms minimum. They are labeled spacing
stress; the other 240 tracks contain 1920 hits. Stress results are separate so
the spacing constraint's predictable misses do not disappear into one average.

| Method | Normal-spacing matches / 1920 | Normal extras | Stress matches / 384 | Stress extras | Noise-only extras |
| --- | --- | --- | --- | --- | --- |
| Fixed | 1091 | 1522 | 132 | 56 | 273 |
| Original adaptive | 638 | 26 | 38 | 0 | 8 |
| Selected adaptive | 1049 | 164 | 62 | 0 | 40 |

The selected setting improves matches over the original adaptive setting but
admits more noise. It substantially reduces extras relative to fixed threshold
10, with slightly fewer matches overall outside spacing stress. At 120 BPM and
four hits/beat, selected adaptive matches only 102/384 hits versus fixed 232/384,
even though nominal spacing is above 100 ms. Neighboring events influence the
local statistics; STFT timing and spacing suppression can also affect close
candidates. The fastest case combines dense-event thresholds with a spacing
constraint, so its misses cannot all be attributed to one cause.

Longer noise controls produce false detections even for settings with zero
extras in the earlier short-fixture study. Noisy matches can still be accidental
within the tolerance, especially in dense tracks. Passing the accounting checks
means counts are consistent, not that rhythm detection or BPM estimation works.
Defaults remain unchanged. Further analysis is needed before treating this
peak-picked event list as a reliable musical beat grid.


## Threshold versus spacing diagnostics

Run `python evals/onset_diagnostics.py --label "Threshold and spacing diagnostic"`.
Suite version 1 reuses all 288 sequence tracks and 36 controls with the same
samples, seeds, frame-center timestamps and 50 ms one-to-one scorer. The selected
threshold remains radius 12 / multiplier 2.5 / floor 0.1. Four branches change
one filter at a time:

| Branch | Threshold | Minimum interval |
| --- | --- | --- |
| All peaks | 0 (strictly positive local peaks) | 0 |
| Threshold only | Selected adaptive | 0 |
| Spacing only | 0 | 100 ms |
| Both | Selected adaptive | 100 ms |

Boundary and plateau rules are active in every branch. Disabling the threshold
also removes the 0.1 floor, so small numerical peaks can enter the candidate
list. This is a mechanism diagnostic, not a proposed detector or parameter
selection procedure. No retuning or default changes occur.

The report saves per-branch indices/times/metrics, threshold-rejected frames,
spacing-suppressed frames, and a trace containing each positive local peak's
strength, threshold and status. Candidate accounting follows the nested path
all peaks -> threshold only -> both. Separately, optimal one-to-one match counts
give unmatched hits before filtering, additional misses after thresholding,
and additional misses after spacing. These increments sum to final misses.
Optimal matching may reassign pairs, so increments are not causal labels for
individual hits. Spacing-only and both-filter survivor sets need not be nested:
rejecting a stronger peak first can let a weaker neighboring peak survive.

All 288 event tracks and 36 controls reproduce the saved selected-adaptive
baseline exactly (input hashes, candidate frames and per-hit metrics).

| Branch | Matches / 2304 | Misses | Extras on event tracks | Noise-only extras |
| --- | --- | --- | --- | --- |
| All peaks | 2304 | 0 | 11382 | 1602 |
| Threshold only | 1111 | 1193 | 164 | 40 |
| Spacing only | 1862 | 442 | 3119 | 546 |
| Both | 1111 | 1193 | 164 | 40 |

Across the 1920 non-stress hits, threshold filtering loses 871 matches; across
384 spacing-stress hits it loses 322. Spacing removes no candidate after this
threshold in either group or in the controls. Thus the threshold, not spacing,
is the active bottleneck on this particular filtered candidate list. This does
not make spacing harmless: without thresholding it reduces matches by 442
(262 non-stress, 180 stress) while removing many extras. Dense hits can raise
the local threshold above the peaks, as illustrated by the clean burst plots.

Raw peaks matching every truth does not mean every event was recognized: noisy
peaks can land within the tolerance by chance, and the raw branch admits over
11000 extra event-track candidates. Removing filters is not a useful fix.
This study uses already-observed synthetic fixtures, not fresh validation data.

Reports are saved as `evals/results/onset-diagnostics-<run-id>.json`; comparison
and peak-stage PNGs are under `outputs/onset_diagnostics/<run-id>/`. Source/input
hashes, settings, environment versions and Git state are retained. Exit status
checks nested sets, candidate partitions, nonnegative match losses and scoring
accounting, not improved accuracy. Tests independently isolate the filters,
check interactions, preserve peak rules, and verify full-branch baseline parity.


## Median/MAD threshold comparison

Run `python evals/onset_robust.py --label "Raw median/MAD comparison"`.
Suite version 1 fixes radius 12, multiplier 4, floor 0.1 before running either
seed group, without searching/tuning. The threshold is local median plus four
times raw MAD, floored at 0.1. No Gaussian scaling factor (such as 1.4826) is
applied. The current frame is included in the centered window, which clips at
edges without padding. This is an offline method.

Comparators are fixed 10, original mean/std radius 8 / multiplier 3, and selected
mean/std radius 12 / multiplier 2.5. All other settings stay fixed: Fs 2048,
STFT 128/hop 32, 100 ms minimum spacing, frame-center times, no energy refinement,
50 ms one-to-one matching. The existing isolated and repeated-event fixtures
and scoring are reused. For an isolated truth, the sequence scorer reduces to
the same nearest-candidate matching rule.

| Group | Seeds | Event tracks | Known hits | Noise controls |
| --- | --- | --- | --- | --- |
| Isolated baseline | 401/503/607 | 576 | 576 | 12 |
| Sequences baseline | 701/809 | 288 | 2304 | 36 |
| Isolated fresh | 1013/1217 | 384 | 384 | 8 |
| Sequences fresh | 1013/1217 | 288 | 2304 | 36 |

Fresh noise seeds do not overlap any prior evaluation's seeds. They are shared
between fixture regimes, and noise is paired within comparisons. Zero-noise
fixtures repeat; new draws are not new signal families or independent tracks.
All three repeated-event baseline arms reproduce the earlier saved candidates,
input hashes and per-hit metrics exactly.

Fresh results below include controls in false counts and F1. Sequence totals
include the 83.3 ms spacing stress; regular/stress summaries are also saved.

| Group / method | Matches | Misses | False including controls | F1 |
| --- | --- | --- | --- | --- |
| Isolated / selected mean/std | 291/384 | 93 | 32 | 0.823 |
| Isolated / median/MAD | 287/384 | 97 | 57 | 0.788 |
| Sequences / selected mean/std | 1091/2304 | 1213 | 240 | 0.600 |
| Sequences / median/MAD | 1328/2304 | 976 | 584 | 0.630 |

On fresh regular-spacing sequences, median/MAD matches 1223/1920 hits versus
1035 for selected mean/std; on stress cases it matches 105/384 versus 56. The
fresh sequence noise controls alone produce 104 candidates for median/MAD
versus 36 for selected mean/std. Thus recovery of dense hits comes with more
noise detections, and the isolated-event comparison is worse on fresh draws.
The clean 120 BPM/four-bursts-per-beat example recovers all eight hits, versus
two with selected mean/std; it is an illustration rather than a general claim.

When local MAD is zero, the threshold becomes max(floor, median). Small values
above a constant background can pass. Sustained/dense activity can still shift
the median. The 100 ms spacing constraint remains active even in the fastest
tracks. Noisy peaks can match by chance, and methods' conditional timing-error
means use different matched populations. These synthetic cases do not validate
real-music performance or justify replacing the existing defaults.

Reports `evals/results/onset-robust-<run-id>.json` preserve per-case indices,
timestamps, match/error details, input/source hashes, all settings, versions,
and Git state. Summaries cover event cases, controls, combined scoring, shapes,
noise levels, and regular/stress cases. Plots in `outputs/onset_robust/<run-id>/`
show comparisons and clean/noisy threshold examples. Tests verify known raw-MAD
values, clipped neighborhoods, zero spread, input validation, gain scaling,
dense-hit recovery, baseline compatibility, and seed separation. Exit status
checks accounting, not improvement. The fresh seeds are now observed data;
future tuning needs another evaluation set.
