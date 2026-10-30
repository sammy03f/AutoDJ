# 🎧 AutoDJ

**An intelligent DJ engine for automatic music mixing.**

AutoDJ analyzes the musical structure of tracks and automatically
constructs smooth, musically-aware transitions between them.

Rather than applying a fixed crossfade, AutoDJ uses digital signal
processing, optimization, and machine learning to reason about:

- tempo and beat alignment
- musical key and harmonic compatibility
- phrase and structural boundaries
- energy progression
- vocal overlap
- spectral characteristics
- transition compatibility

The long-term goal is simple:

> Give AutoDJ a collection of songs and let it build the mix.

## How it works

Songs
  ↓
Audio Analysis
  ↓
Beat / Key / Energy / Structure
  ↓
Transition Scoring
  ↓
Playlist Optimization
  ↓
DSP Transition Engine
  ↓
Continuous Mix

## Project Status

Early development

The first audio component loads uncompressed 16-bit PCM WAV files and exposes
their sample rate, channels, duration, and integer samples. Time ranges can be
selected from loaded audio, adjusted in amplitude, saved as WAV, and plotted
as waveform images. A small educational DFT now extracts frequency coefficients
and is checked against NumPy's FFT. Frequency spectrum PNGs show the amplitudes
of those components, with optional Hann windowing to reduce distant leakage.
A numerical STFT follows frequency content over time and can be saved as an
amplitude spectrogram. Spectral flux now measures positive spectral changes
as a first onset-strength baseline, with thresholded peak picking for candidate
onsets, plus optional energy-based timing refinement. A repeatable volume/noise
evaluation now measures misses and false detections. A small tuning study has
selected a better local mean/std threshold on separate noise seeds, while keeping
defaults unchanged. Repeated-event evaluations now expose dense-hit suppression
and false detections on longer noise controls. A filter-stage diagnostic identifies
the selected threshold as the active bottleneck in those fixtures. An optional
median/MAD threshold recovers more sequence hits but admits more noise; it has
not replaced the existing settings. A first global tempo candidate now comes
from autocorrelation of the full onset-strength curve, with explicit half-time
and nonrhythmic-input limitations. A hop-size comparison resolves the tested
clean half-time error at finer resolution, without changing defaults. Beat
tracking is not yet implemented.

## Development

Requires Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import autodj; print(autodj.__file__)"
```

Matplotlib draws waveform images; NumPy provides our FFT reference. The install
command above installs these dependencies. Run the tests from the activated
virtual environment with:

```sh
python -m unittest discover -s tests -v
```

Load a local 16-bit PCM WAV file:

```python
from autodj.audio import load_wav

audio = load_wav("assets/local/song.wav")
print(audio.sample_rate, audio.channels, audio.duration)
print(audio.samples[:5])  # First five frames, one integer per channel.
```

Extract seconds 10–15 from a track at least 15 seconds long:

```python
from autodj.audio import select_time_range

clip = select_time_range(audio, 10.0, 15.0)
print(clip.duration)
```

The start frame is included and the end frame is excluded. Times convert to
frame indices using `floor(seconds * sample_rate)`; an endpoint equal to the
track duration includes all remaining frames. Equal times produce an empty
clip. Negative, reversed, nonfinite, or out-of-bounds times raise `ValueError`.
The original audio is unchanged.

Adjust a clip's amplitude:

```python
from autodj.audio import apply_gain

quieter = apply_gain(clip, 0.5)  # Half the amplitude.
louder = apply_gain(clip, 2.0)   # Twice the amplitude, with clipping if needed.
```

Gain multiplies every sample equally, preserving duration and channels. Zero
produces silence; one preserves the samples. Results round to integers (ties
to even) and clamp to the 16-bit range, -32768 to 32767. Clamping is clipping
and can cause distortion. Negative or nonfinite gains raise `ValueError`.
The original clip is unchanged. Half the amplitude does not necessarily sound
half as loud.

Save the adjusted clip as a 16-bit PCM WAV file:

```python
from pathlib import Path
from autodj.audio import save_wav

Path("outputs").mkdir(exist_ok=True)
save_wav(quieter, "outputs/clip.wav")
```

The output keeps the sample rate, channels, and exact integer samples. The
parent directory must exist; saving overwrites an existing file at that path.
Invalid metadata, frame shapes, or sample values raise `ValueError` before
the destination is opened. Both loading and saving build audio data in memory.

Save a waveform image (the `outputs/` directory must exist):

```python
from autodj.audio import save_waveform, select_time_range

# A short excerpt makes individual oscillations easier to see.
detail = select_time_range(audio, 0, min(0.02, audio.duration))
save_waveform(detail, "outputs/waveform.png", title="First 20 milliseconds")
```

The PNG has one plot per channel, with time in seconds and raw PCM amplitude.
All plots use the same amplitude scale so quieter audio looks smaller and
clipping remains visible. Extracted clips start at zero on the time axis.
Every sample is plotted; use short excerpts for detail and to limit memory use.
Existing PNGs are overwritten. Empty audio produces an annotated empty plot.
No GUI window is opened.

Generate original, quieter, clipped, and stereo examples:

```sh
python examples/waveforms.py
```

Open the PNGs in `outputs/waveforms/` side by side to compare them. Rerunning
the example replaces these images.

Try the educational DFT on a tiny single-channel signal:

```python
from autodj.dsp import dft

spectrum = dft([0, 1, 0, -1])
print(spectrum)  # Approximately (0, -2j, 0, 2j).
```

For loaded audio, select one channel explicitly, for example
`dft([frame[0] for frame in audio.samples[:64]])`. This direct implementation
does O(N²) work: use tiny excerpts, not entire tracks. Output is unnormalized
complex coefficients, including negative-frequency bins. Empty or nonfinite
inputs raise `ValueError`. See [DSP notes](docs/math.md) for the formula and scaling.

Run the two-tone example and save a repeatable FFT comparison report:

```sh
python examples/dft.py
python evals/dft.py --label "Initial DFT baseline"
```

The example finds 220 Hz and 440 Hz tones with amplitudes 1 and 0.5. Evaluation
reports are retained under `evals/results/dft-*.json`.

Save a frequency spectrum for a short, single-channel sequence:

```python
from autodj.audio import save_spectrum

save_spectrum([0, 1, 0, -1], 8000, "outputs/spectrum.png")
```

This example has a 2000 Hz peak of amplitude 1. For loaded audio, use
`save_spectrum([frame[0] for frame in audio.samples[:256]], audio.sample_rate,
"outputs/spectrum.png")`. Samples must be nonempty, finite, and real.
The parent directory must exist; an existing PNG is overwritten.

The plot shows nonnegative frequencies through Nyquist in Hz and amplitude in
the input's units (PCM values stay in PCM units). It scales DFT magnitudes to
recover bin-centered sinusoid amplitudes, treating DC and Nyquist separately.
Off-bin tones spread across bins, so their tallest bin may underestimate their
amplitude. The display auto-scales vertically; read the axis values when comparing
images. This still uses our O(N²) DFT, so keep inputs short.

Generate the two-tone image, with peaks at 220 Hz and 440 Hz:

```sh
python examples/spectrum.py
```

Open `outputs/spectra/two_tones.png`. For numeric results without plotting, use
`frequencies, amplitudes = amplitude_spectrum(samples, sample_rate)` from
`autodj.dsp`.

Apply a periodic Hann window when computing or plotting a spectrum:

```python
from autodj.dsp import amplitude_spectrum

frequencies, amplitudes = amplitude_spectrum(samples, sample_rate, window="hann")
save_spectrum(samples, sample_rate, "outputs/hann.png", window="hann")
```

`window=None` remains the default. Hann multiplies the samples by a smooth
taper before the DFT; division by the sum of the weights compensates for its
amplitude loss. The input is unchanged. It reduces distant leakage while
broadening the main peak, and cannot fully recover off-bin tone amplitudes.

Compare 220 Hz (aligned with a bin) and 230 Hz (between bins):

```sh
python examples/windowing.py
```

Each run saves a comparison image and window-shape image under a new
`outputs/windowing/<run-id>/` directory, plus a numeric observation report in
`evals/results/windowing-<run-id>.json`. The comparison uses fixed axes and both
linear amplitude and dB relative to amplitude 1; its displayed dB floor does not
change the calculated spectrum. See [DSP notes](docs/math.md) for the formula.

Compute an STFT for one channel:

```python
from autodj.dsp import stft

analysis = stft(
    [frame[0] for frame in audio.samples], audio.sample_rate,
    frame_size=1024, hop_size=256,
)
print(analysis.coefficients.shape)  # (number of frames, 513 frequency bins)
print(analysis.times[:5])          # Frame START times in seconds.
print(analysis.frequencies[:5])    # Frequency bins in Hz.
```

Each complete frame gets a periodic Hann window and an FFT. The hop controls
how many samples the next frame advances; it must be between 1 and frame size.
There is no centering or padding, and an incomplete final frame is omitted.
Empty input or input shorter than one frame returns zero rows. The returned
coefficients are raw complex values, preserving phase; they are not yet an
amplitude plot. Inputs must be finite, real, and single-channel. This version
keeps input and output in memory.

Run the known tone-switch example:

```sh
python examples/stft.py
```

It tracks a switch from 64 Hz to 192 Hz at 0.25 seconds, checks against the
direct DFT, and saves a timestamped JSON report under `evals/results/` plus
NumPy arrays under `outputs/stft/`. The frame crossing the switch contains both
tones. The example also saves `spectrogram.png` beside its arrays and records
the image path and display settings in the report.
See [DSP notes](docs/math.md) for timing, shape, and amplitude conventions.

Plot the STFT from your loaded audio (create `outputs/` first):

```python
from autodj.audio import save_spectrogram

save_spectrogram(analysis, "outputs/spectrogram.png", reference_amplitude=32768)
```

Use reference amplitude 32768 for raw 16-bit PCM, or the default 1 for synthetic
or normalized audio. Color shows gain-corrected amplitude in dB relative to
that reference, with a fixed default range of -80 to 0 dB. Silence reaches the
floor; levels outside the display range saturate. This is not a power-density
or acoustic-loudness plot, and the strongest point is not automatically made 0 dB.

The horizontal axis uses frame starts. Each column occupies one hop interval;
its underlying analysis window may extend farther. The right edge is the last
frame start plus one hop, not necessarily the end of the audio. Frequency cells
are centered on FFT bins, bounded by 0 Hz and Nyquist. Empty STFT results raise
`ValueError`. Saving overwrites existing PNGs without opening a GUI.

Compute an onset-strength curve from an STFT:

```python
from autodj.analysis import spectral_flux

strength = spectral_flux(analysis)
# strength[m] corresponds to analysis.times[m] (frame start).
```

This sums positive changes in linear FFT magnitude between consecutive frames.
The first value is zero because there is no preceding frame. Empty STFTs return
an empty curve. Input arrays are preserved. Values use raw FFT units and depend
on signal gain, frame size, and hop size; they are not probabilities or detected
onset times. The strength calculation does not smooth or threshold the curve.

Select candidate peaks with explicit settings:

```python
from autodj.analysis import pick_onsets

candidates = pick_onsets(strength, analysis.times, threshold=10, min_interval=0.1)
print(candidates.frame_indices)
print(candidates.times)  # Uncorrected frame-start estimates, in seconds.
```

These settings are for the synthetic example, not universal music defaults.
Peaks must be strictly above the threshold and have lower neighbors on both
sides. Flat tops use their earliest frame; boundary peaks are excluded. When
peaks are closer than `min_interval` seconds, the stronger wins; ties favor the
earlier one. Accepted peaks return in time order. Exactly the minimum interval
is allowed. The picker preserves the supplied timestamps without correction.

Generate known bursts and plot their strength curve:

```sh
python examples/onsets.py
```

Each run retains a PNG and NumPy arrays in `outputs/onsets/<run-id>/` plus a JSON
report in `evals/results/onsets-<run-id>.json`. Dashed lines mark known burst
starts; crosses mark selected candidates. A frame can span a burst even when
its start precedes it. With this fixture, candidates occur 31.25 ms early.
Suite version 2 reports one-to-one matches, misses, extra candidates, and signed
timing error using a declared 31.25 ms tolerance. This evaluates the synthetic
baseline, not real-music accuracy.

Refine candidate times with a short trailing energy curve:

```python
from autodj.analysis import short_time_energy, refine_onset_times

energy = short_time_energy(samples, window_size=8)
centers = candidates.times + analysis.frame_size / (2 * analysis.sample_rate)
refined_times = refine_onset_times(
    energy, analysis.sample_rate, centers,
    search_radius=analysis.frame_size / analysis.sample_rate,
    relative_threshold=0.1,
)
```

The window is in samples (8 samples is about 3.9 ms at 2048 Hz). The function
looks for an energy rise within the search region and retains the supplied time
if no crossing is found. It preserves candidate count and correspondence;
overlapping searches can yield duplicate or reordered timestamps. It does not
recover missed events or enforce minimum spacing again.

Compare frame-start, frame-center, and refined timing across synthetic cases:

```sh
python evals/onset_timing.py --label "Timing baseline"
```

The evaluation covers clicks, sharp bursts, and slower attacks across three
frame sizes, three hop ratios, and four event offsets. Each run retains a report
in `evals/results/onset-timing-*.json` and a plot in `outputs/onset_timing/`.
On the initial 108-case sweep, 100 cases produced one candidate and 8 produced
none. For those 100 paired cases, frame-center mean absolute error was 11.25 ms
versus 40.60 ms for frame starts, but 12 slow-attack cases became less accurate.
Timing summaries exclude missed/multiple detections; reports count them separately
and retain all cases in tolerance-score denominators. Public timing defaults
remain unchanged. Suite version 2 adds energy refinement: mean absolute error
is 3.14 ms on the same 100 paired cases. Compared with centers, 84 improve,
7 worsen, and 9 tie. The 8 misses remain. A second plot shows the energy crossing
on a slow attack. These isolated signals have silent backgrounds; this is not
a real-music accuracy claim. See [evaluation details](evals/README.md).

Evaluate detection at different event volumes and background noise levels:

```sh
python evals/onset_sensitivity.py --label "Fixed threshold baseline"
```

This keeps the detector unchanged and runs 576 event cases plus 12 noise-only
controls with seeded Gaussian noise. Reports under `evals/results/` retain
matches, misses, false detections, precision/recall, and conditional timing
errors for frame-start, frame-center, and energy-refined timestamps. Plots are
saved under `outputs/onset_sensitivity/`.

The baseline misses every quiet (gain 0.1) event in silence. At noise standard
deviation 0.2, noise-only controls produce 6.67 false detections per second.
Energy refinement can follow unrelated noise rises and reduces matches from
403 to 342 out of 576 at the declared 50 ms tolerance. High noisy-case recall
can include chance matches; inspect precision and controls alongside recall.
See [evaluation details](evals/README.md) for settings and limitations.

Try an optional threshold that follows the local onset-strength background:

```python
from autodj.analysis import local_onset_threshold, pick_onsets

thresholds = local_onset_threshold(strength, radius=8, std_multiplier=3, floor=0.1)
candidates = pick_onsets(strength, analysis.times, threshold=thresholds, min_interval=0.1)
```

The threshold is `max(floor, local mean + multiplier * local standard deviation)`.
Radius is in frames; the centered neighborhood includes the current frame and
clips at signal edges. This uses future frames and is an offline method.
The existing scalar threshold API still works; timestamps and peak-spacing rules
are unchanged. Candidate timestamps above remain frame starts.

Compare fixed/adaptive detection with common frame-center timestamps:

```sh
python evals/onset_adaptive.py --label "Local mean plus 3 std baseline"
```

Each seed group contains 576 event cases plus 12 noise-only controls. On fresh
seeds, adaptive detection gives 262 matches, 314 misses, and zero extras, versus
360 matches, 216 misses, and 603 extras with the fixed threshold. The adaptive
rule suppresses noise but also real events, so it remains optional. JSON reports
and comparison/threshold plots retain both successes and failures. Parameters
are fixed before evaluation; see [evaluation details](evals/README.md).

Search a small parameter grid with separate tuning and evaluation noise seeds:

```sh
python evals/onset_threshold_search.py --label "Threshold selection baseline"
```

The study maximizes tuning F1 across event cases and noise-only controls, freezes
the chosen setting, then evaluates it on new noise draws. It selected radius 12
and multiplier 2.5 with floor 0.1. On 576 held-out event cases it matched 425,
missed 151, and produced zero false detections, including controls (F1 0.849).
The original adaptive setting matched 263; fixed threshold 10 matched 402 with
649 false detections including controls. This is evidence for this synthetic
noise distribution, not a general music benchmark.

To try the selected setting explicitly:

```python
thresholds = local_onset_threshold(strength, radius=12, std_multiplier=2.5, floor=0.1)
candidates = pick_onsets(strength, analysis.times, threshold=thresholds, min_interval=0.1)
```

Public defaults remain unchanged. Reports retain all tuning scores and held-out
case details under `evals/results/onset-threshold-search-*.json`, with a comparison
plot under `outputs/onset_threshold_search/`.

Evaluate frozen settings on repeated click and burst sequences:

```sh
python evals/onset_sequences.py --label "Repeated events baseline"
```

The suite generates 288 eight-hit tracks at known tempos, plus 36 duration-matched
noise-only controls. BPM is an input to the generator, not an estimated result.
One-to-one matching prevents a candidate from counting for multiple hits. Reports
retain per-hit errors, misses, extras, and separate spacing-stress results.

Outside the 180 BPM/four-hits-per-beat stress cases, the selected adaptive rule
matches 1049/1920 hits with 164 extra candidates, versus 1091 matches and 1522
extras for fixed threshold 10. Dense sequences can raise the adaptive threshold
and suppress real hits. The selected method also produces 40 candidates on the
longer noise-only controls. This extends the isolated-event findings without
retuning or changing defaults. Reports are saved under `evals/results/` and plots
under `outputs/onset_sequences/`.

Separate threshold rejection from minimum-spacing suppression:

```sh
python evals/onset_diagnostics.py --label "Threshold and spacing diagnostic"
```

Four branches keep the same strength curve and peak rules: all positive local
peaks, selected threshold only, 100 ms spacing only, and both filters. The full
branch reproduces the saved sequence baseline. Reports retain rejected frame
indices, per-peak strengths/thresholds, match losses and false detections.

On these fixtures, threshold-only and both-filters results are identical:
1111/2304 hits matched, 1193 missed, and 164 extra candidates. Spacing removes
no candidate after the selected threshold, although spacing alone reduces
matches when threshold filtering is disabled. All positive local peaks can
match every truth within 50 ms but also produce 11382 extras, so that result
is not a usable detector. The next threshold investigation should address dense
events raising the local background estimate while still rejecting noise.
Defaults are unchanged. Reports and plots are under `evals/results/` and
`outputs/onset_diagnostics/`.

Try an optional local median/MAD threshold:

```python
from autodj.analysis import robust_onset_threshold

thresholds = robust_onset_threshold(strength, radius=12, mad_multiplier=4, floor=0.1)
candidates = pick_onsets(strength, analysis.times, threshold=thresholds, min_interval=0.1)
```

MAD is `median(abs(local_values - local_median))`. The threshold is
`max(floor, local_median + multiplier * MAD)`; MAD is unscaled. The centered
window includes the current frame and clips at edges, so it uses future frames.
A zero MAD leaves the threshold at the local median or floor, which can admit
small background fluctuations.

```sh
python evals/onset_robust.py --label "Raw median/MAD comparison"
```

The settings are fixed before evaluating known and fresh noise seeds. On fresh
repeated sequences, median/MAD improves matches from 1091 to 1328 out of 2304
compared with the selected mean/std setting, but false detections including
controls rise from 240 to 584. F1 rises from 0.600 to 0.630. On fresh isolated
events F1 falls from 0.823 to 0.788. This is a tradeoff, not a new default.
Reports and comparison/threshold plots are saved under `evals/results/` and
`outputs/onset_robust/`; see [evaluation details](evals/README.md).

Estimate a global tempo candidate directly from onset strength:

```python
from autodj.analysis import estimate_tempo

# Strength has one value per STFT hop, not per audio sample.
frame_rate = analysis.sample_rate / analysis.hop_size
result = estimate_tempo(strength, frame_rate, min_bpm=60, max_bpm=200)
print(result.bpm)  # None when no supported positive correlation is available.
print(result.lag_frames, result.score)
```

This uses the full curve, without onset thresholds or timestamp refinement.
It subtracts the mean and compares delayed copies: `BPM = 60 * frame_rate / lag`.
The result retains searched integer lags and normalized autocorrelation scores.
A score is not confidence: even noise or tiny steady-tone numerical fluctuations
can produce a tempo candidate. Silence and constant strength return no estimate.
No beat timestamps, octave correction, or streaming analysis are provided.

```sh
python evals/tempo.py --label "First tempo baseline"
```

The first run gets 32/40 clean steady rhythm cases within 3% of nominal BPM.
All eight 150 BPM cases return about 75.29 BPM (half-time). Alternating accents
and equal subdivisions are evaluated separately because their musical pulse is
ambiguous. Silence and a single burst return no tempo in the controls, while
steady tone and noise return spurious candidates. Reports retain all outcomes
and score curves under `evals/results/tempo-*.json`; comparison and autocorrelation
plots are under `outputs/tempo/`. This is an educational baseline, not a reliable
song BPM detector yet.

Compare tempo accuracy and timing cost at smaller STFT hops:

```sh
python evals/tempo_hops.py --label "Hop resolution comparison"
```

This processes the same 40 steady rhythms, 20 ambiguity cases, and four controls
with hops 32/16/8, while keeping the frame size at 128 and the estimator unchanged.
The strength rates are 64/128/256 frames per second. Both smaller hops get 40/40
steady cases within 3%; the 150 BPM cases return 150.59 instead of 75.29 BPM.
Ambiguity cases improve from 6/20 to 10/20, but half/double-time interpretation
and spurious control estimates remain unresolved.

This run's median full-clip processing times were 4.64/8.13/15.12 ms, using
three repeats after warm-up. These are diagnostic timings, not performance
guarantees. A smaller hop changes both the flux curve and the lag grid, so this
is not a pure test of lag rounding. Reports and score/runtime plots are saved
under `evals/results/tempo-hops-*.json` and `outputs/tempo_hops/`. Existing
examples and defaults remain unchanged; hop 16 is a candidate for further tests.

Run the repeatable synthetic evaluation cases and save a report plus audio:

```sh
PYTHONPATH=src python3 evals/audio_io.py --label "WAV saving baseline"
```

Each run adds a JSON report under `evals/results/` and listening examples under
`outputs/evals/`. See [evaluation history](evals/README.md) for cases, metrics,
and how to compare runs. Reports can be committed; generated WAV files stay local.

This initial loader reads the whole file into memory as Python tuples; use small
files while exploring. Other bit depths and floating-point WAV are unsupported.
Keep local input audio in
`assets/local/` and generated audio in `outputs/`; both are ignored by Git.
Public examples must use appropriately licensed audio.

## Project documentation

- [Full project context and engineering instructions](docs/project-context.md)
- [Architecture](docs/architecture.md)
- [Roadmap and next task](docs/roadmap.md)
- [Mathematics and DSP notes](docs/math.md)

## Roadmap

- [x] Audio loading and waveform analysis
- [x] FFT / STFT / spectrogram generation
- [ ] Beat and tempo detection
- [ ] Beat matching and time stretching
- [ ] Automatic crossfades
- [ ] Phrase and structure detection
- [ ] Musical key detection
- [ ] Harmonic compatibility scoring
- [ ] Vocal-aware transitions
- [ ] EQ and filter-based DJ transitions
- [ ] Automatic transition-point selection
- [ ] Playlist sequencing
- [ ] Energy-curve optimization
- [ ] Audio embeddings
- [ ] ML transition-quality model
- [ ] Personalized transition preferences
