# Architecture

AutoDJ will analyze, sequence, and mix locally supplied audio. The current
implementation supports 16-bit PCM WAV loading, time selection, amplitude gain,
and WAV saving.

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
the input. `evals/audio_io.py` runs fixed synthetic cases and retains JSON reports
in `evals/results/`; generated audio stays in ignored `outputs/evals/` directories.
Introduce analysis, DSP, transitions, sequencing, and ML modules only as their
features are implemented. Prefer simple, testable functions and explicit data
over speculative abstractions. No runtime dependencies are needed yet.

Build an explainable deterministic baseline before adding learned models.
Infrastructure and the product interface come after a working audio engine.

The complete supplied project brief is in [project-context.md](project-context.md).
