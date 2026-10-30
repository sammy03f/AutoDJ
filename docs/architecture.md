# Architecture

AutoDJ will analyze, sequence, and mix locally supplied audio. The current
implementation supports 16-bit PCM WAV loading, time selection, amplitude gain,
WAV saving, waveform PNG export, an educational direct DFT, spectrum PNG export,
numerical STFT analysis, amplitude spectrogram PNGs, spectral-flux onset strength,
candidate onset peak picking, optional energy-based timing refinement, and an
experimental global tempo candidate from onset-strength autocorrelation.

The intended flow is:

```text
Audio loading → Audio analysis → Track representation
                                      ↓
                       Transition scoring and sequencing
                                      ↓
                            Transition selection
                                      ↓
                             DSP rendering → Mix
```

Keep three responsibilities separate:

- Analysis describes what happens in the music.
- Decision-making chooses track order, transition points, and techniques.
- Rendering manipulates audio to perform those decisions.

Code lives in `src/autodj/`. `audio/loader.py` reads WAV files into `WavAudio`,
which holds the sample rate, channel count, and integer samples grouped by frame.
`audio/writer.py` validates and encodes those samples back into 16-bit PCM WAV.
`audio/segments.py` and `audio/gain.py` return transformed audio without changing
the input. `audio/waveform.py` uses Matplotlib's noninteractive Agg renderer to
save one amplitude-versus-time plot per channel. It plots all samples on a fixed
16-bit PCM scale; this first version is intended for short clips.
`evals/audio_io.py` runs fixed synthetic cases and retains JSON reports
in `evals/results/`; generated audio stays in ignored `outputs/evals/` directories.
`dsp/fourier.py` computes the direct DFT of a single-channel sequence using the
standard library. It is intentionally O(N²) for learning on short signals.
`dsp/spectrum.py` computes one-sided frequency/amplitude pairs for real input;
`audio/spectrum.py` draws those values with Matplotlib. The numerical calculation
is tested independently of rendering, including DC, Nyquist, and odd lengths.
`dsp/windows.py` implements the periodic Hann formula without another dependency.
The spectrum functions accept `window="hann"` and correct amplitude by dividing
by the sum of the weights. Default rectangular-window behavior is preserved.
`dsp/stft.py` uses NumPy's real FFT per complete Hann-windowed frame. `StftResult`
retains raw complex coefficients shaped (frames, bins), frame-start times,
frequencies, framing settings, and window sum. It does not pad, center frames,
or normalize magnitudes. The input and result reside in memory.
`audio/spectrogram.py` consumes `StftResult`, corrects amplitudes by the window
sum, and renders fixed-reference dB colors without altering coefficients. Its
time cells begin at frame starts and span one hop; this is a display convention,
not the duration of each analysis window. Matplotlib draws explicit cell edges.
NumPy now supplies STFT computation as well as independent FFT comparisons.
`analysis/onsets.py` consumes `StftResult` and returns one spectral-flux value per
frame. It uses raw linear magnitudes, ignores decreases, and preserves STFT
frame-start alignment. Plotting and synthetic fixture generation live in
`examples/onsets.py`. A separate `pick_onsets` function takes strength and times,
applies a scalar or per-frame threshold and local-maximum rule, then suppresses close peaks
in descending strength order. `OnsetCandidates` holds frame indices and unchanged
timestamps. This offline greedy baseline does not estimate BPM or refine timing.
`analysis/energy.py` computes a trailing mean-square curve at sample resolution.
`refine_onset_times` in `analysis/onsets.py` optionally uses this curve to locate
an upward energy crossing within a bounded neighborhood of each candidate.
It returns a separate timestamp array, keeping candidate correspondence without
changing the original frame indices or enforcing spacing/order after refinement.
`evals/onset_timing.py` compares frame-start, frame-center, and refined times on
the same selected events, saving metrics and comparison/energy plots. Refinement
does not recover missed detections or change the peak picker's default timing.
`evals/onset_sensitivity.py` reuses the timing suite's clean synthetic fixtures
and adds paired, seeded Gaussian noise and event gain changes. It independently
scores each timestamp method against one known event per case, and scores
noise-only controls separately. This evaluation adds no production algorithm
or default changes; reports preserve all candidate timestamps and source hashes.
`local_onset_threshold` computes an inspectable centered mean/std threshold
curve in `analysis/onsets.py`. It uses clipped neighborhoods including the current
frame. `pick_onsets` accepts this curve or its existing scalar threshold; both
paths share peak, plateau, and spacing rules. `evals/onset_adaptive.py` reuses the
sensitivity fixtures/scorer and compares fixed/adaptive thresholds using identical
frame-center timestamps, without refinement. Seed groups and parameters are
recorded separately, and full threshold/strength curves accompany the metrics.
`evals/onset_threshold_search.py` calls the same evaluator with explicit
radius/multiplier settings. It ranks a fixed grid on tuning seeds, freezes the
selection, then evaluates only the selected setting and original baselines on
new seeds. The objective includes false detections from noise controls. It saves
all tuning scores, fixture hashes, and held-out case details without modifying
production defaults or using held-out scores to select parameters.
`evals/onset_sequences.py` tests the frozen threshold configurations on eight-hit
click/burst tracks and duration-matched noise controls. Its evaluator uses
chronological one-to-one dynamic-programming matching (maximum matches, then
minimum total absolute error), with per-hit match indices and missed/false
indices retained. Spacing stress is reported separately. Sequence generation,
matching and plotting are evaluation code; no tempo estimator or production
threshold changes are introduced.
`evals/onset_diagnostics.py` reuses the sequence fixtures and one-to-one scorer.
It calls the production peak picker in four branches with threshold and spacing
independently enabled/disabled, retaining plateau/boundary rules. Selected
threshold parameters stay frozen. Candidate set differences identify rejected
and suppressed frames; match-count differences along nested candidate sets
measure lost matching capacity without assigning causal labels to individual
truth events. The evaluator records controls and all four branches; it does
not modify the production detector.
`robust_onset_threshold` in `analysis/onsets.py` computes a separate median/raw-MAD
curve using the same centered, clipped neighborhood convention. It feeds the
existing peak picker without changing its rules or the mean/std helper.
`evals/onset_robust.py` compares fixed, original mean/std, selected mean/std and
median/MAD settings on both isolated and repeated fixtures with known and fresh
noise seeds. All methods share frame-center timestamps, spacing and scoring;
controls and spacing-stress summaries remain separate. No automatic threshold
selection, energy refinement, or production default change is introduced.
`analysis/tempo.py` consumes a uniformly sampled strength curve and its frame
rate (audio sample rate / hop size). `estimate_tempo` searches integer lags
inside explicit BPM bounds, with at least two period lengths of input per lag.
It scales to avoid overflow, subtracts the mean, and normalizes delayed dot
products by zero-lag energy. `TempoEstimate` contains the selected BPM/lag/score
(or None) and all searched lags/scores. The score is not calibrated confidence.
This path does not use thresholded candidates or refined timestamps and does
not return a beat grid. `evals/tempo.py` independently generates known rhythms,
reports octave errors and nonrhythmic controls, and saves score-curve plots.
`evals/tempo_hops.py` reuses the tempo fixtures and control signal generator,
keeping the waveform and frame size fixed while varying hop size. It measures
STFT/flux and tempo time separately, repeats each case, and records full lag
curves plus diagnostic nominal/half-time bins. Ground truth only determines
scoring and probes; it never influences the estimator. The original hop-32
branch reproduces all saved numerical baseline results. This experiment changes
neither production algorithms nor default/example settings.
`estimate_tempo` also accepts opt-in raw-peak and normalized-correlation floors.
Both default to zero, preserving prior numerical results. Rejections retain
None for BPM/lag/score and add `TempoEstimate.rejection_reason`. Reasons are
insufficient_data, no_variation, weak_strength, no_positive_correlation, or
low_correlation; accepted results have no rejection reason. Early rejects leave
the searched score array zero; low-correlation rejects retain the computed
curve. `evals/tempo_evidence.py` compares baseline, each check separately and
both together on quiet/noisy rhythms and varied controls. It preserves the
unfiltered candidate and curve alongside every decision for inspection.
`evals/tempo_diagnostics.py` reuses those controls at four gain multipliers and
adds six clean rhythm references. It records raw and relative peak flux, full
and half-window lag curves, and agreement separately from missing estimates.
The halves slice the existing flux curve without resetting its boundary;
neighboring STFT frames still overlap in audio samples. These measurements
introduce no production acceptance rule or default changes.
`evals/tempo_relative.py` selects a relative-flux floor using tuning cases only,
then generates a separate parameter/seed evaluation split. The experimental
decision replaces the absolute-strength floor and retains the existing .2
correlation floor. It never changes the selected BPM, and retains original
results alongside the new decisions. Reports include the full threshold search,
split definitions, input/source hashes, score curves, and per-family outcomes.
The rule currently lives only in evaluation code.
Introduce further analysis, DSP, transitions, sequencing, and ML modules only as their
features are implemented. Prefer simple, testable functions and explicit data
over speculative abstractions. Matplotlib is the first runtime dependency,
used for drawing plots; PCM loading, editing, and saving use the standard library.

Build an explainable deterministic baseline before adding learned models.
Infrastructure and the product interface come after a working audio engine.

The complete supplied project brief is in [project-context.md](project-context.md).
