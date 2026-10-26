# Architecture

AutoDJ will analyze, sequence, and mix locally supplied audio. The current
implementation supports 16-bit PCM WAV loading, time selection, amplitude gain,
WAV saving, waveform PNG export, an educational direct DFT, spectrum PNG export,
and numerical STFT analysis.

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
NumPy now supplies STFT computation as well as independent FFT comparisons.
Introduce further analysis, DSP, transitions, sequencing, and ML modules only as their
features are implemented. Prefer simple, testable functions and explicit data
over speculative abstractions. Matplotlib is the first runtime dependency,
used for drawing plots; PCM loading, editing, and saving use the standard library.

Build an explainable deterministic baseline before adding learned models.
Infrastructure and the product interface come after a working audio engine.

The complete supplied project brief is in [project-context.md](project-context.md).
