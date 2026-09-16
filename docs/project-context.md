# AutoDJ — Project Context and Engineering Instructions

## 1. Project Overview

We are building **AutoDJ**, an intelligent DJ system that automatically analyzes, sequences, and mixes music.

The long-term goal is:

> A user uploads a collection of songs, chooses a desired mix style or energy trajectory, and AutoDJ automatically determines the best song order, identifies musically appropriate transition points, beat-matches the tracks, chooses an appropriate transition strategy, and renders a continuous DJ mix.

This is NOT intended to be a simple crossfade application.

The project should eventually combine:

- software engineering
- digital signal processing (DSP)
- applied mathematics
- optimization
- graph algorithms
- machine learning
- recommendation/personalization concepts
- music/audio analysis

The project is also educational. I want to deeply understand the mathematics, algorithms, DSP, ML, and engineering behind every major component rather than simply wrapping existing libraries.

Do not prematurely implement the entire system.

We will build AutoDJ incrementally, with each stage producing something that works and can be tested before moving to the next stage.


# 2. Core Product Experience

Eventually, the user experience should look approximately like this:

1. User uploads multiple audio tracks.
2. AutoDJ analyzes each track.
3. AutoDJ extracts musical/audio features.
4. AutoDJ calculates compatibility between tracks.
5. AutoDJ determines an appropriate playlist ordering.
6. AutoDJ identifies candidate transition regions.
7. AutoDJ scores those transitions.
8. AutoDJ chooses the best transition points.
9. AutoDJ chooses an appropriate transition technique.
10. AutoDJ beat-matches the tracks.
11. AutoDJ renders the transitions.
12. AutoDJ produces one continuous mix.

Eventually, users should also be able to specify preferences such as:

- Smooth
- Club
- Aggressive
- Chill
- Workout
- Build to peak
- Peak then cooldown

The system should be capable of optimizing the resulting set around those preferences.


# 3. High-Level Architecture

Conceptually, the system is:

Audio Files
    |
    v
Audio Loading
    |
    v
Audio Analysis
    |
    +--> waveform
    +--> spectrum
    +--> BPM
    +--> beat grid
    +--> downbeats
    +--> bars
    +--> phrases
    +--> musical structure
    +--> musical key
    +--> energy
    +--> loudness
    +--> spectral characteristics
    +--> vocal activity
    +--> audio embeddings (later)
    |
    v
Track Representation
    |
    +-------------------------+
    |                         |
    v                         v
Transition Scoring      Playlist Sequencing
    |                         |
    +------------+------------+
                 |
                 v
          Transition Selection
                 |
                 v
          DSP Transition Engine
                 |
        +--------+---------+
        |        |         |
        v        v         v
    tempo      EQ       crossfade
    matching  filters      etc.
        |
        v
        Continuous Mix


# 4. Repository Architecture

Use a clean Python package structure similar to:

autodj/
|
|-- README.md
|-- LICENSE
|-- .gitignore
|-- pyproject.toml
|
|-- docs/
|   |-- architecture.md
|   |-- roadmap.md
|   |-- math.md
|
|-- src/
|   |-- autodj/
|       |-- __init__.py
|       |
|       |-- audio/
|       |   |-- __init__.py
|       |   |-- loader.py
|       |   |-- waveform.py
|       |   |-- spectrogram.py
|       |
|       |-- analysis/
|       |   |-- __init__.py
|       |   |-- tempo.py
|       |   |-- beats.py
|       |   |-- key.py
|       |   |-- energy.py
|       |   |-- vocals.py
|       |   |-- structure.py
|       |
|       |-- dsp/
|       |   |-- __init__.py
|       |   |-- filters.py
|       |   |-- timestretch.py
|       |   |-- equalizer.py
|       |   |-- crossfade.py
|       |
|       |-- transitions/
|       |   |-- __init__.py
|       |   |-- scorer.py
|       |   |-- selector.py
|       |   |-- renderer.py
|       |
|       |-- sequencing/
|       |   |-- __init__.py
|       |   |-- graph.py
|       |   |-- optimizer.py
|       |
|       |-- ml/
|           |-- __init__.py
|           |-- embeddings.py
|           |-- dataset.py
|           |-- model.py
|
|-- tests/
|
|-- examples/
|
|-- assets/


Do not create unnecessary infrastructure just because it may eventually be useful.

In particular, do not prematurely introduce:

- microservices
- Kubernetes
- complex cloud infrastructure
- distributed systems
- databases
- frontend frameworks
- message queues
- ML pipelines

Those can be introduced when actual requirements justify them.


# 5. Engineering Philosophy

This project should follow several important principles.

## Principle 1 — Understand before abstracting

Important DSP and mathematical concepts should be understood before relying completely on libraries.

For educationally important algorithms, we may:

1. implement a simple version ourselves,
2. test it,
3. compare it against an established implementation,
4. then use the production-quality implementation when appropriate.

Examples include:

- DFT
- spectral analysis
- onset detection
- tempo estimation
- filters
- crossfades
- similarity metrics


## Principle 2 — DSP before ML

Do NOT immediately introduce machine learning.

First construct a deterministic DSP/algorithmic baseline.

For example:

TransitionScore =
    tempo compatibility
    + harmonic compatibility
    + phrase alignment
    + energy compatibility
    + vocal compatibility
    + spectral compatibility

Only after this works should ML attempt to improve it.

We want to eventually be able to answer:

> Does the learned model actually outperform the deterministic baseline?


## Principle 3 — ML must solve a real problem

Do not add AI simply to make the project sound AI-powered.

ML should eventually solve problems where handcrafted rules are insufficient, such as:

- transition-quality prediction
- learned audio similarity
- musical structure recognition
- vocal detection
- personalized transition ranking
- user preference modeling


## Principle 4 — Decisions should be explainable

AutoDJ should eventually be able to explain why it selected a transition.

Example:

Selected transition:
Song A @ 2:48
Song B @ 0:31

Tempo compatibility:      96%
Harmonic compatibility:   91%
Phrase alignment:         100%
Energy continuity:        87%
Vocal compatibility:      94%

Overall score:             94%

This is important both for debugging and for understanding the system.


## Principle 5 — Separate analysis, decision-making, and rendering

These are different responsibilities.

Analysis asks:

> What is happening in the music?

Decision-making asks:

> What should the DJ do?

Rendering asks:

> How do we manipulate the audio to perform that decision?

Do not tightly couple these systems.


# 6. Development Roadmap


## Phase 0 — Digital Audio Fundamentals

Goal:

Understand how digital audio is represented.

Topics:

- waveform
- samples
- sample rate
- amplitude
- stereo/mono channels
- bit depth
- PCM
- clipping
- quantization
- Nyquist frequency
- aliasing

Initial functionality:

- load WAV audio
- inspect metadata
- access raw samples
- select time ranges
- manipulate amplitude
- visualize waveform
- save modified audio

The relationship

sound <-> numerical samples

should be completely understood before proceeding.


## Phase 1 — Frequency Analysis

Goal:

Understand the frequency-domain representation of audio.

Topics:

- sinusoids
- frequency
- phase
- complex numbers
- Fourier transform
- DFT
- FFT
- windowing
- STFT
- spectrograms

Implement a simple DFT ourselves:

X[k] = sum from n=0 to N-1 of:

x[n] * exp(-i * 2*pi*k*n/N)

Then compare against:

numpy.fft.fft

We should understand why the naive DFT is O(N^2) and why FFT algorithms are substantially faster.

Then implement/use STFT to represent frequency over time.

Deliverable:

Given a track, generate:

- waveform visualization
- frequency spectrum
- spectrogram


## Phase 2 — Beat and Tempo Analysis

Goal:

Determine where beats occur and estimate tempo.

Topics:

- transients
- onsets
- onset strength
- spectral flux
- autocorrelation
- periodicity
- BPM
- beat tracking

Conceptually:

If beats occur every T seconds:

BPM = 60 / T

Deliverable:

Input:
song.wav

Output:

Estimated BPM: 127.8

Beat timestamps:
0.483
0.952
1.421
1.891
...

Visualize detected beats over the waveform.

Where practical, compare our algorithm with an established implementation such as librosa.


## Phase 3 — Beat Matching

Goal:

Make two songs play at compatible tempos.

Example:

Song A = 128 BPM
Song B = 124 BPM

Required stretch ratio:

r = 128 / 124

Study:

- resampling
- interpolation
- pitch
- tempo
- time stretching
- STFT
- phase vocoder concepts

Understand why naive resampling changes both tempo and pitch.

Eventually use a production-quality time-stretching method, while understanding the underlying DSP.

Deliverable:

Take two tracks with different tempos and create tempo-compatible versions without obviously incorrect pitch changes.


## Phase 4 — First Automatic Transition

Goal:

Produce the first real DJ output.

Input:

songA.wav
songB.wav

Initially, transition points may be manually specified.

Implement:

- linear crossfade
- equal-power crossfade

Linear:

y(t) = (1-a(t))A(t) + a(t)B(t)

Equal power:

y(t) = cos(theta(t))A(t) + sin(theta(t))B(t)

Deliverable:

Two tracks go in.

One beat-aligned transition comes out.

This is the first major product milestone.


## Phase 5 — Musical Structure

Goal:

Make transitions aware of how music is organized.

Understand:

beats
-> downbeats
-> bars
-> phrases
-> sections

Possible sections include:

- intro
- verse
- chorus
- buildup
- drop
- breakdown
- outro

Transitions should generally be evaluated around musically meaningful boundaries rather than arbitrary timestamps.

This also reduces the transition search space.


## Phase 6 — Musical Key and Harmonic Mixing

Goal:

Estimate harmonic compatibility between tracks.

Learn:

- pitch
- pitch classes
- notes
- major/minor scales
- intervals
- circle of fifths
- harmonic mixing
- chroma features

Represent pitch classes roughly as:

[C, C#, D, D#, E, F, F#, G, G#, A, A#, B]

Estimate key and create a harmonic compatibility function:

K(A, B) -> [0, 1]

The system should use harmonic compatibility as one signal when evaluating possible transitions.


## Phase 7 — DJ Transition Engine

Goal:

Move beyond simple crossfades.

Implement transition techniques such as:

- smooth crossfade
- equal-power crossfade
- bass swap
- EQ blend
- low-pass filter sweep
- high-pass filter sweep
- loop transition
- drop transition
- aggressive transition

Learn:

- digital filters
- FIR
- IIR
- cutoff frequency
- Q factor
- frequency response
- equalization

For bass swaps, avoid having two incompatible bass lines dominate simultaneously.

Conceptually:

Song A bass:
HIGH -> LOW

Song B bass:
LOW -> HIGH


## Phase 8 — Automatic Transition Selection

Goal:

Allow AutoDJ to determine WHEN two tracks should transition.

For tracks A and B, candidate transitions are:

(A, t_A) -> (B, t_B)

Features may include:

- BPM difference
- key compatibility
- energy difference
- phrase alignment
- vocal overlap
- spectral similarity
- loudness difference

Initially create an interpretable deterministic scoring function:

S =
    w1 * tempo_score
  + w2 * harmonic_score
  + w3 * energy_score
  + w4 * phrase_score
  + w5 * vocal_score
  + w6 * spectral_score

Then choose:

(t_A*, t_B*) = argmax S(A, t_A, B, t_B)

The scoring system should expose component scores for debugging and explainability.


## Phase 9 — Playlist Sequencing

Goal:

Given many songs, determine a good order automatically.

Represent tracks as a weighted directed graph.

Each track is a vertex.

Edge:

A -> B

has weight:

w(A,B) = transition compatibility from A into B

Then find a path through the tracks that maximizes overall transition quality.

Objective:

maximize sum of:

w(S_i, S_(i+1))

Explore algorithms such as:

- greedy search
- dynamic programming for small cases
- beam search
- local search
- simulated annealing
- other appropriate combinatorial optimization techniques

Do not blindly implement a complicated algorithm before understanding the problem.


## Phase 10 — Energy-Curve Optimization

Allow the user to define the desired shape of the DJ set.

Examples:

- gradually increasing
- workout
- smooth/chill
- build -> peak -> cooldown
- consistently high energy

Represent desired energy:

E_target(t)

and actual playlist energy:

E_actual(t)

One possible objective component is:

sum over t of:

(E_actual(t) - E_target(t))^2

Playlist optimization can combine:

transition quality + energy trajectory.

For example:

J =
lambda_1 * transition_quality
- lambda_2 * energy_error


## Phase 11 — Audio Embeddings and Machine Learning

Only after the deterministic system works should we introduce learned models.

Represent audio segments using embeddings:

z_A in R^d
z_B in R^d

Similarity may initially use cosine similarity:

similarity(A,B) =
(z_A dot z_B) / (||z_A|| ||z_B||)

Potential models:

- logistic regression baseline
- tree-based model
- small neural network
- learned audio embeddings
- ranking models

Eventually predict:

P(good transition | A, t_A, B, t_B)

The learned system should be evaluated against the deterministic baseline.


## Phase 12 — Transition Quality Dataset

We need a legitimate way to evaluate transitions.

Possible strategy:

Generate candidate transitions and collect human ratings.

Example:

Transition 382:
A -> B

Rating:
1 2 3 4 5

Potential training sample:

x_i = transition features
y_i = human rating/preference

Pairwise ranking may also be useful:

Given Transition A and Transition B:

Which sounds better?

A / B

This may be more reliable than absolute numerical ratings.

Do not scrape or redistribute copyrighted commercial music.

Public demos should use appropriately licensed audio.


## Phase 13 — Personalization

Eventually learn individual user preferences.

Possible preferences:

- aggressive vs smooth
- short vs long transitions
- harmonic consistency
- tolerance for tempo jumps
- preferred energy progression
- vocal overlap tolerance

Represent a user's preferences with:

u in R^d

Eventually estimate:

P(user likes transition | u, z_A, z_B, transition_features)

This connects AutoDJ to recommendation and personalization systems.


## Phase 14 — Productization

Only once the audio engine works well should we build a serious product layer.

Potential architecture:

React / TypeScript frontend
        |
        v
Python API
        |
        v
Analysis / Mixing Workers
        |
        +--> DSP
        +--> ML
        +--> Optimization
        |
        v
Audio / Feature Storage

At that point we can evaluate whether we actually need:

- FastAPI
- asynchronous job processing
- database
- caching
- object storage
- background workers
- frontend
- deployment infrastructure

Infrastructure should solve real requirements rather than being added for resume keywords.


# 7. Testing Philosophy

Testing is important from the beginning.

Different components need different forms of testing.

Examples:

Audio loader:
- correct sample rate
- correct channel count
- correct duration
- correct sample shape

DFT:
- compare with NumPy FFT on small deterministic signals

Tempo:
- synthetic click tracks with known BPM

Filters:
- synthetic sine waves with known frequencies

Crossfades:
- verify length
- verify boundaries
- verify gain curves
- detect clipping

Transition scoring:
- deterministic unit tests

Playlist optimization:
- small graphs with known optimal solutions

ML:
- train/validation/test separation
- baseline comparisons
- reproducibility

For DSP algorithms, synthetic signals should be used whenever possible because their ground truth is known.


# 8. Documentation Philosophy

The repository should explain not just WHAT the code does but WHY.

Maintain:

docs/architecture.md
docs/roadmap.md
docs/math.md

math.md should eventually contain explanations of important mathematical concepts used by AutoDJ, such as:

- sampling
- Nyquist theorem
- DFT
- FFT
- STFT
- convolution
- autocorrelation
- filters
- cosine similarity
- optimization objectives
- graph formulations
- ML loss functions

This should be written in my own understanding as the project develops.


# 9. Code Quality

Code should generally be:

- typed where useful
- modular
- documented
- testable
- readable
- deterministic where possible

Avoid:

- giant files
- giant functions
- unexplained constants
- premature abstractions
- unnecessary classes
- hidden global state
- excessive dependencies
- copy-pasted algorithms without explanation

Prefer simple functions until the domain clearly requires more complex abstractions.


# 10. Git Philosophy

Use meaningful commits.

Examples:

chore: initialize AutoDJ project structure

feat(audio): add WAV loading and metadata extraction

feat(audio): add waveform visualization

feat(dsp): implement discrete Fourier transform

feat(dsp): add FFT-based spectral analysis

feat(analysis): implement onset detection

feat(analysis): add tempo estimation

feat(transitions): implement equal-power crossfade

test(analysis): validate tempo estimator against synthetic click tracks

Avoid meaningless commit messages such as:

update
stuff
fix
changes


# 11. Milestones

Suggested milestones:

v0.1 — Hear the Music
- load audio
- waveform
- FFT
- STFT
- spectrogram

v0.2 — Find the Beat
- onset detection
- BPM
- beat grid
- beat visualization

v0.3 — First Transition
- tempo matching
- beat alignment
- crossfade
- render two-song mix

v0.4 — Understand the Music
- downbeats
- phrases
- key
- energy
- vocals
- structure

v0.5 — AutoMix
- candidate transitions
- compatibility scoring
- automatic transition selection
- multiple transition strategies

v0.6 — Build the Set
- track graph
- playlist optimization
- energy trajectory

v0.7 — Learn to DJ
- embeddings
- transition dataset
- learned transition ranking
- baseline evaluation

v0.8 — Personal DJ
- user feedback
- preference learning
- personalization

v1.0 — AutoDJ
- polished end-to-end experience
- multiple tracks
- automatic sequencing
- automatic transition selection
- high-quality rendered mix
- explainability
- demo


# 12. Important Copyright/Data Constraint

Do not design the system around downloading audio from Spotify or scraping copyrighted tracks.

For development:

- use audio files supplied locally by the user
- use synthetic audio for testing
- use appropriately licensed/open music for public examples

Do not commit local commercial music files to the Git repository.

Generated audio and local input directories should be ignored by Git where appropriate.


# 13. How Codex Should Work With Me

This section is especially important.

I am building this project partly to learn DSP, applied mathematics, ML, and audio engineering.

Do not treat me as someone who only wants the finished code.

When implementing a major new algorithm:

1. Explain what problem we are solving.
2. Explain the relevant mathematical/CS concept.
3. Explain the proposed implementation.
4. Identify what we should implement ourselves versus use a library for.
5. Keep the first implementation as simple as reasonably possible.
6. Add tests.
7. Let me understand the component before jumping several milestones ahead.

Do not silently build future phases unless I explicitly request them.

If I ask for a specific issue, work primarily on that issue.

If a design decision will significantly affect later architecture, explain the tradeoff before making a large change.

When using a library for an important algorithm, tell me what the library is doing conceptually.

If there is a simple educational implementation and a production implementation, we may intentionally keep both.

For example:

estimate_tempo_simple(...)
estimate_tempo_reference(...)

This lets us understand and evaluate our implementation.


# 14. Current State

The project is at the very beginning.

The immediate objective is NOT machine learning.

The immediate objective is to build the audio/DSP foundation correctly.

The first development sequence should be:

1. clean repository/package setup
2. WAV/audio loading
3. metadata extraction
4. waveform visualization
5. DFT implementation
6. FFT comparison
7. STFT
8. spectrogram
9. onset detection
10. tempo estimation
11. beat tracking
12. two-track beat alignment
13. equal-power crossfade
14. render the first automatic two-song transition

That first working transition is our initial major target.


# 15. Definition of Success

The final project should allow me to demonstrate something like:

"Give AutoDJ these songs."

AutoDJ analyzes them.

It determines:

- BPM
- beat grid
- key
- energy
- musical structure
- vocal activity
- spectral/audio similarity

It determines a playlist ordering.

It chooses transition locations.

It explains:

"Song A -> Song B was selected because the tempo, key, phrase boundary,
energy progression, and vocal structure make this a strong transition."

It then performs:

- tempo matching
- beat alignment
- EQ/filter manipulation
- crossfading
- other appropriate DJ transition techniques

and produces one continuous mix.

The system should eventually combine:

DSP
+ applied mathematics
+ algorithms
+ optimization
+ ML
+ personalization
+ production software engineering

But every layer should exist because it improves the actual DJ system, not simply because it looks impressive on a resume.


# 16. What To Do Next

Before changing anything, inspect the existing repository.

Report:

1. the current repository structure,
2. which pieces of the desired structure already exist,
3. anything that should be reorganized,
4. the smallest set of changes needed to establish the clean AutoDJ foundation,
5. the proposed first implementation task.

Do NOT implement the entire roadmap.

After inspecting the repository, begin with the earliest unfinished milestone unless I explicitly ask for something else.