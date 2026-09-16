# Architecture

AutoDJ will analyze, sequence, and mix locally supplied audio. The current
implementation contains only the Python package foundation.

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

Code lives in `src/autodj/`. Add `audio/` with the first WAV-loading task;
introduce analysis, DSP, transitions, sequencing, and ML modules only as their
features are implemented. Prefer simple, testable functions and explicit data
over speculative abstractions. No runtime dependencies are needed yet.

Build an explainable deterministic baseline before adding learned models.
Infrastructure and the product interface come after a working audio engine.

The complete supplied project brief is in [project-context.md](project-context.md).
