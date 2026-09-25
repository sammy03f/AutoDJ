# Roadmap

## Current work

- [x] Establish a minimal Python package and documentation foundation.
- [x] Load 16-bit PCM WAV audio and inspect sample rate, channels, duration, and samples.
- [x] Select time ranges from loaded audio.
- [x] Adjust amplitude with gain and 16-bit clipping.
- [ ] Save audio as WAV.
- [ ] Visualize waveforms.
- [ ] Implement a small educational DFT and compare it with NumPy FFT.
- [ ] Generate an STFT and spectrogram.

Completed first component: a standard-library WAV loader for uncompressed
16-bit PCM, with synthetic fixtures checking samples, channel layout, metadata,
and invalid inputs. Samples remain signed integers grouped into frames.

Time selection now converts seconds to frame indices, preserving channels and
samples. Amplitude adjustment multiplies samples by a nonnegative gain, rounds
to integers, and clips to the 16-bit PCM range. Next small task: save modified
audio as WAV so extracted and adjusted clips can be heard.

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
