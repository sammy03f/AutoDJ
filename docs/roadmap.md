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
- [ ] Select candidate onset timestamps from the strength curve.

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
Next small task: select candidate onset timestamps, with explicit peak-selection
and timing rules. Tempo estimation follows later.

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
