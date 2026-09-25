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
their sample rate, channels, duration, and integer samples. Next: time-range
selection and amplitude adjustment, followed by waveform and frequency analysis.

## Development

Requires Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import autodj; print(autodj.__file__)"
```

There are no runtime dependencies. Run the synthetic WAV tests with:

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
