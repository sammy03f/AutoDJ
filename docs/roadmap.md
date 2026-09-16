# Roadmap

## Current work

- [x] Establish a minimal Python package and documentation foundation.
- [ ] Load WAV audio and inspect sample rate, channels, duration, and samples.
- [ ] Select time ranges, adjust amplitude, and save audio.
- [ ] Visualize waveforms.
- [ ] Implement a small educational DFT and compare it with NumPy FFT.
- [ ] Generate an STFT and spectrogram.

First implementation task: WAV loading and metadata extraction. Explain PCM
samples, sample rate, and channel layout before implementation. Use synthetic
WAV fixtures to verify sample values, shape, sample rate, channel count, and
duration. Decide the initial supported PCM formats explicitly.

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
