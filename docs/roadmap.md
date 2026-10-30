# Roadmap

## Current work

- [x] Establish a minimal Python package and documentation foundation.
- [x] Load 16-bit PCM WAV audio and inspect sample rate, channels, duration, and samples.
- [x] Select time ranges from loaded audio.
- [x] Adjust amplitude with gain and 16-bit clipping.
- [x] Save audio as WAV.
- [x] Save repeatable synthetic audio evaluation results and listening examples.
- [x] Visualize waveforms with separate channel plots and synthetic examples.
- [x] Implement a small educational DFT and compare it with NumPy FFT.
- [x] Plot a frequency spectrum and explain bin spacing and amplitude scaling.
- [x] Explore spectral leakage and periodic Hann windowing with gain correction.
- [x] Compute an STFT with explicit frame timing and tone-switch tests.
- [x] Plot an STFT as an amplitude spectrogram.
- [x] Compute spectral-flux onset strength and test known synthetic transients.
- [x] Select candidate onset timestamps with explicit peak and spacing rules.
- [x] Compare frame-start and frame-center timing across transient shapes and alignments.
- [x] Add optional energy-based onset timing refinement and evaluate it.
- [x] Evaluate detection sensitivity across gain and background noise.
- [x] Implement and evaluate an optional local mean/std onset threshold on baseline and fresh seeds.
- [x] Study the adaptive threshold tradeoff with separate tuning and evaluation noise seeds.
- [x] Evaluate frozen thresholds on repeated-event sequences and longer noise controls.
- [x] Diagnose threshold rejection versus spacing suppression on the frozen sequence suite.
- [x] Implement and evaluate an optional median/MAD threshold with noise controls and fresh seeds.
- [x] Build and evaluate a first autocorrelation tempo baseline from unthresholded onset strength.
- [x] Compare STFT hop sizes to diagnose the clean-tempo half-time error and processing cost.
- [ ] Address spurious tempo candidates on nonrhythmic inputs before beat tracking.

Completed first component: a standard-library WAV loader for uncompressed
16-bit PCM, with synthetic fixtures checking samples, channel layout, metadata,
and invalid inputs. Samples remain signed integers grouped into frames.

Time selection now converts seconds to frame indices, preserving channels and
samples. Amplitude adjustment multiplies samples by a nonnegative gain, rounds
to integers, and clips to the 16-bit PCM range. WAV saving completes the
load → select → gain → save workflow. Synthetic evaluations retain JSON reports
and local listening examples; see [evaluation history](../evals/README.md).
Waveform PNGs now show each channel against time using a fixed PCM amplitude
scale. `examples/waveforms.py` illustrates attenuation, clipping, and stereo.
`dsp/fourier.py` now implements the direct DFT, with known-answer tests and a
NumPy FFT comparison. `examples/dft.py` identifies two synthetic tones;
`evals/dft.py` saves numerical comparison reports.
`dsp/spectrum.py` converts the DFT to a one-sided amplitude spectrum, and
`audio/spectrum.py` saves the plot. `examples/spectrum.py` shows the expected
220 Hz and 440 Hz peaks with amplitudes 1 and 0.5.
`dsp/windows.py` adds a periodic Hann taper. Spectra optionally use it with
coherent-gain correction. `examples/windowing.py` compares 220/230 Hz tones and
saves plots and numeric observations for later comparison.
`dsp/stft.py` applies Hann and a real FFT to complete overlapping frames.
It returns complex coefficients with frame-start times and frequency bins.
`examples/stft.py` tracks a 64 Hz → 192 Hz switch and saves reference comparisons.
`audio/spectrogram.py` renders gain-corrected one-sided amplitudes in dB with
explicit frame-start coordinates and a fixed reference. The STFT example saves
a PNG alongside its arrays and report. The v0.1 audio-analysis foundation is in place.
`analysis/onsets.py` sums positive frame-to-frame FFT magnitude changes without
normalization or smoothing. `examples/onsets.py` plots known bursts and retains
their curve, arrays, and validation report.
`pick_onsets` now selects local maxima above an absolute threshold and suppresses
nearby weaker peaks. The burst example reports candidate frame starts, one-to-one
matches, and timing error; this baseline is 31.25 ms early on the current fixture.
`evals/onset_timing.py` compares 108 isolated-event cases and 18 negative controls.
Center timestamps reduce mean absolute error on the 100 single-candidate cases
from 40.60 ms to 11.25 ms, but worsen 12 slow-attack cases; 8 cases are missed
by the fixed threshold. Public timing defaults remain frame starts.
`short_time_energy` and `refine_onset_times` now provide an optional local
energy-rise estimate. Suite version 2 reduces paired mean absolute error to
3.14 ms: 84 cases improve versus center labels, 7 worsen, and 9 tie. All 8 misses
remain. Short-window smoothing and gradual attacks still bias the crossing late;
background noise and competing sounds are not covered by this isolated suite.
`evals/onset_sensitivity.py` now holds detection settings fixed across 576 event
cases and 12 noise-only controls. Frame-center scoring matches 403/576 events
with 588 extra candidates; refined scoring matches 342/576 with 649 extras
at 50 ms tolerance. Gain-0.1 events in silence are all missed; the strongest
noise-only condition produces 6.67 false detections per second. These results
expose fixed-threshold and refinement weaknesses without changing production.
`local_onset_threshold` now supplies a per-frame curve to `pick_onsets`, whose
scalar threshold behavior remains available. The first rule uses radius 8,
mean + 3 population standard deviations, and floor 0.1, with no parameter search.
`evals/onset_adaptive.py` evaluates baseline seeds and fresh seeds 101/211/307.
On fresh seeds, extras fall from 603 to zero, but matches fall from 360 to 262
out of 576 events. The rule recovers some quiet events but suppresses many
broader peaks, so it is optional rather than a new default.
`evals/onset_threshold_search.py` selects among 12 radius/multiplier settings
using tuning seeds 101/211/307 and F1 including noise-only false detections.
Radius 12 / multiplier 2.5 / floor 0.1 wins before any analysis of held-out
seeds 401/503/607. On those draws it matches 425/576 events with 151 misses
and zero false detections, versus 263 matches for the original adaptive rule
and 402 matches with 649 false detections for the fixed rule (including controls).
Defaults stay unchanged. Remaining misses occur in noisy conditions; quiet
clean events are recovered. These are new noise draws, not new signal families.
`evals/onset_sequences.py` now evaluates 288 eight-hit tracks (2304 known hits)
and 36 noise-only controls. One-to-one matching maximizes matched hits, then
minimizes total timing error. Normal-spacing cases and 83.3 ms spacing stress
cases are reported separately. On normal-spacing cases, selected adaptive
matches 1049/1920 hits with 164 extras, versus fixed 1091 with 1522 extras.
Selected adaptive also produces 40 false candidates on the longer controls.
Dense events can raise their own local threshold, and the 100 ms spacing rule
cannot preserve every hit in the fastest sequences. Defaults remain unchanged.
`evals/onset_diagnostics.py` now toggles threshold and spacing independently.
The full branch exactly reproduces the sequence baseline. Threshold-only and
full results are identical on all event tracks and controls: selected-threshold
filtering accounts for 1193 additional unmatched hits; spacing suppresses zero
surviving candidates. Spacing alone does reduce matches when thresholding is
disabled, but that branch admits 3119 extras. Candidate and match-count effects
are saved separately; noisy raw peaks can match truth by chance.
`robust_onset_threshold` now provides local median + 4 raw MAD, radius 12,
floor 0.1 as an optional alternative. `evals/onset_robust.py` compares all four
methods on isolated and repeated events, known seeds and fresh seeds 1013/1217.
Fresh sequence matches improve from 1091 to 1328/2304 versus selected mean/std,
but false candidates including controls rise from 240 to 584. Fresh isolated
F1 worsens from 0.823 to 0.788. No existing threshold settings are replaced.
`estimate_tempo` now computes a mean-subtracted autocorrelation over supported
integer lags and returns a global tempo candidate with the searched score curve.
`evals/tempo.py` evaluates 40 clean steady tracks, 20 accent/subdivision challenges,
and 4 controls. Primary accuracy is 32/40 within 3%; all eight 150 BPM cases
return half-time near 75.29 BPM. The ambiguity set yields 6 correct, 6 half-time,
2 double-time, and 6 other estimates. Silence and a single burst abstain; steady
tone and noise produce spurious candidates, so scores are not confidence.
No onset settings change and no beat grid is produced.
`evals/tempo_hops.py` now compares hops 32/16/8 on identical fixtures, fixed
frame size 128 and an unchanged estimator. Hops 16 and 8 get 40/40 steady cases
within 3% (150 BPM returns 150.59 rather than 75.29); ambiguity cases reach
10/20 versus 6/20. Median clip processing times in the first run are
4.64/8.13/15.12 ms with three repetitions after warm-up. Full scores and probes
near nominal/half-time lags are saved. Hop changes affect flux as well as lag
resolution, so this does not isolate rounding alone. Defaults remain unchanged.
Next small task: address spurious tempo candidates on steady tones and noise.
Finer hops do not solve these cases or musical half/double-time ambiguity.

## Subsequent milestones

| Version | Deliverable |
| --- | --- |
| v0.1 | Audio loading, waveform, FFT, STFT, spectrogram |
| v0.2 | Onsets, tempo, beat grid, beat visualization |
| v0.3 | Tempo matching, beat alignment, first rendered two-track transition |
| v0.4 | Downbeats, phrases, key, energy, vocals, structure |
| v0.5 | Explainable automatic transition selection and multiple techniques |
| v0.6 | Playlist sequencing and energy-curve optimization |
| v0.7 | Transition dataset, embeddings, learned ranking evaluated against baseline |
| v0.8 | Feedback and personalization |
| v1.0 | Polished end-to-end automatic mixes with explanations |

Finish and understand each stage before moving on. The first major target is
a working two-song transition. See [project-context.md](project-context.md)
for the complete phased plan and engineering instructions.
