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
as a first onset-strength baseline. Next: select candidate onsets from that curve.

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
onset times. No smoothing, thresholding, or tempo estimation is applied yet.

Generate known bursts and plot their strength curve:

```sh
python examples/onsets.py
```

Each run retains a PNG and NumPy arrays in `outputs/onsets/<run-id>/` plus a JSON
report in `evals/results/onsets-<run-id>.json`. Dashed lines mark the known burst
starts, not detections. A frame can span a burst even when its start timestamp
precedes it, so the curve may rise before those markers.

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

- [ ] Audio loading and waveform analysis
- [ ] FFT / STFT / spectrogram generation
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
