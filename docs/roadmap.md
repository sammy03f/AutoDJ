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
- [ ] Investigate a noise-aware onset threshold against the saved sensitivity baseline.

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
Next small task: investigate a noise-aware threshold, checking both false
positives and misses against this baseline. Tempo estimation follows later.

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
