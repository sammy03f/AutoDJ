# Mathematics and DSP notes

Develop these notes alongside working components, recording explanations in
the project owner's own understanding.

## Amplitude and gain

For each sample x, a constant gain g produces y = g * x. Our first version
accepts finite g >= 0: zero gives silence, one preserves amplitude, and two
doubles it. Every channel receives the same gain; timing stays unchanged.

For example, samples (1000, -2000) with gain 0.5 become (500, -1000).
Because our audio stores integers, fractional results round to the nearest
integer, with ties going to the even integer: 1.5 becomes 2 and 0.5 becomes 0.
This rounding introduces quantization error.

16-bit signed PCM can represent only -32768 through 32767. A sample of 20000
with gain 2 would become 40000, so we clamp it to 32767. This is hard clipping:
it changes the waveform and can cause audible distortion. Gain describes
amplitude, not perceived loudness; half the amplitude need not sound half as loud.

## Waveform plots

A waveform places each sample's amplitude on the vertical axis and its time
on the horizontal axis. Sample n occurs at t = n / sample_rate seconds.
For example, four frames at 4 Hz occur at 0, 0.25, 0.5, and 0.75 seconds.
Their total duration is 1 second; the last sample is one frame before the end.
Each channel shares those timestamps but has its own amplitude values.

Our plot uses raw integer PCM amplitudes and the same vertical scale for every
channel and image. Automatically scaling each image independently would make
half-gain audio look just as tall as the original. Dashed lines mark -32768
and 32767; a heavily clipped sinusoid has flattened peaks at those limits.

Connecting lines are visual guides between discrete samples, not a reconstruction
of the continuous sound. Short excerpts show individual oscillations clearly.
An extracted clip starts at time zero; it does not retain its original offset.

## Discrete Fourier transform (DFT)

The waveform tells us how amplitude changes with time. The DFT describes a
finite sequence using N equally spaced frequency bins:

```text
X[k] = sum over n=0,...,N-1 of x[n] * exp(-2*pi*i*k*n/N)
```

Here x[n] is a sample, N is the sample count, k is the frequency-bin index,
and i is the imaginary unit (written `1j` in Python). Euler's identity,
`exp(i*theta) = cos(theta) + i*sin(theta)`, lets one complex coefficient
capture both cosine and sine contributions. Its magnitude `abs(X[k])` measures
strength; its angle describes phase. Phase is not meaningful at zero magnitude.

For example, `[0, 1, 0, -1]` is one sine cycle in four samples. Its DFT is
approximately `[0, -2j, 0, 2j]`. The first bin is DC (the sum of the samples),
and the two nonzero bins represent the positive and negative frequency pair.
Tiny residuals near zero are normal floating-point roundoff.

With sample rate Fs, bin spacing is Fs/N Hz. Positive-frequency bin k represents
k*Fs/N Hz. For N=64 at Fs=1760 Hz, bins 8 and 16 are 220 and 440 Hz.
The bins beyond the positive-frequency region represent negative frequencies:
their frequency is (k-N)*Fs/N. For even N, bin N/2 is the Nyquist frequency,
Fs/2, shared by the positive and negative interpretations. For odd N, the last
positive bin is (N-1)/2. Real signals have conjugate positive/negative pairs.

Our forward transform is unnormalized. For a real sinusoid exactly aligned
with a bin, amplitude is `2*abs(X[k])/N` in the positive-frequency region.
DC and the even-N Nyquist bin use `abs(X[k])/N` without doubling. Frequencies
between bins spread across multiple bins (spectral leakage). The direct DFT
itself applies no window or zero padding; the spectrum helper now offers Hann.

`dft` uses two explicit loops: N output bins times N input samples, or O(N²)
work. FFT algorithms compute the same DFT in O(N log N) work. Our implementation
uses the standard library; NumPy provides the independent FFT comparison.
Known-answer tests check magnitude, phase, and normalization before comparison
with NumPy on real and complex signals, including odd lengths.

Reference convention: [NumPy's DFT documentation](https://numpy.org/doc/stable/reference/routines.fft.html).

## From DFT coefficients to a spectrum plot

`amplitude_spectrum(samples, sample_rate)` returns frequencies and amplitudes
for bins 0 through N//2. For real input, the negative-frequency bins carry
conjugate copies, so we combine their magnitudes into the positive side.
The amplitude multiplier is 2/N except for DC and the even-N Nyquist bin,
where it is 1/N. With odd N there is no Nyquist bin: the last retained bin
still has a partner and must be doubled. With N=1, only DC remains.

Worked example: at 1760 Hz with N=64, frequency spacing is 27.5 Hz.
A unit-amplitude 220 Hz sine gives |X[8]|=32, so its plotted amplitude is
2*32/64=1. A 440 Hz sine of amplitude 0.5 gives |X[16]|=16, so its plotted
amplitude is 2*16/64=0.5. These are linear amplitudes, not power or decibels.
Raw PCM input retains PCM amplitude units; it is not scaled to [-1, 1].

The plot draws a vertical line at each discrete frequency bin and auto-scales
the vertical display range. It does not normalize peaks, add frequency bins,
or apply a window by default. Exact tone amplitudes in this example rely on
frequencies lining up with bins.

## Spectral leakage and Hann windowing

The DFT describes a segment as if it repeats. At 1760 Hz with 64 samples,
220 Hz makes exactly 8 cycles, but 230 Hz makes about 8.36 cycles. Repeating
that off-bin segment generally creates a discontinuity at the join. Its spectrum
spreads into surrounding bins: this is leakage, not extra tones in the source.

We multiply samples by a periodic Hann window before the DFT:

```text
w[n] = 0.5 - 0.5*cos(2*pi*n/N), n = 0,...,N-1
windowed[n] = samples[n] * w[n]
```

This form omits the repeated zero endpoint. A four-point window is
`(0, 0.5, 1, 0.5)`. Our single-point convention is `(1,)`; zero/negative or
noninteger lengths are rejected. The periodic form is appropriate for spectral
analysis; a symmetric Hann instead uses N-1 in the denominator and includes
both zero endpoints. [SciPy documents both forms](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html).

For N>1 the average weight, called coherent gain, is 0.5. To compensate,
replace division by N in the amplitude spectrum with division by sum(w).
Keep the existing doubling rules for interior positive-frequency bins.
A unit-amplitude, isolated, bin-centered tone away from DC/Nyquist then retains
its peak amplitude. The correction is for amplitudes, not a power-density estimate.

Hann lowers distant leakage but spreads the main peak more widely. It does not
change bin spacing or remove off-bin amplitude bias (scalloping loss). Nearby
tones or positive/negative lobes near DC/Nyquist can overlap, limiting amplitude
estimates. Windowing does not recover information lost by taking a short segment.

`examples/windowing.py` compares both tones with identical axes. Its log plots
use `20*log10(amplitude / 1)`, with a display floor of -120 dB. Reports measure
the largest amplitude more than two bins (55 Hz) from the known tone, excluding
the wider Hann main lobe. This is a diagnostic for these fixtures, not total
leaked energy or a general audio-quality score. Numeric outputs are not floored.
Tests compare weights with NumPy and windowed spectra with an independent FFT.

## Short-time Fourier transform (STFT)

A single spectrum loses information about when frequencies occur. The STFT
analyzes successive short segments. For frame length L, hop H, and frame index m:

```text
start_sample = m * H
frame[n] = x[m*H + n] * w[n], n = 0,...,L-1
X[m,k] = sum(frame[n] * exp(-2*pi*i*k*n/L))
```

We implement framing and reuse our periodic Hann; NumPy's real FFT computes
the transform efficiently. For S input samples, the complete-frame count is
`max(0, 1 + floor((S-L)/H))`. We require 1 <= H <= L. No padding or centering
is applied. Coefficients have shape `(frame_count, L//2 + 1)`.

The result's `times[m]` is m*H/Fs, the **start** of the frame. Its time support
is [m*H/Fs, (m*H+L)/Fs), and its last sample occurs one sample interval before
that end. Frequencies are k*Fs/L. For L=64, H=32, Fs=1024, frames last 0.0625 s,
start every 0.03125 s, and bins are spaced 16 Hz apart. Longer frames offer
finer bin spacing but mix events over a longer interval. Smaller hops give
more time positions without changing the frame length or frequency spacing.

The output is unnormalized complex coefficients, including phase. To obtain
one-sided amplitudes later, divide abs(coefficients) by `window_sum` and double
interior bins, excluding DC and the even-L Nyquist bin. Do not apply that
doubling to the stored complex coefficients or confuse them with power.

In our example, 512 samples switch from 64 Hz (amplitude 1) to 192 Hz
(amplitude 0.5) at sample 256. There are 15 frames. Frame 7 starts at sample
224 and spans both tones, so it should not be treated as a pure-tone frame.
Tests check the pure frames separately and compare small frames against our
direct DFT. Empty/short signals yield zero frames with the frequency axis intact.

Reference: [NumPy real FFT conventions](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html).

## Displaying the STFT as a spectrogram

Our image uses one-sided amplitude rather than power density. First compute
`A = abs(coefficients) / window_sum`, doubling interior bins but excluding DC
and even-frame Nyquist. With odd frame lengths, the last bin is doubled too.
Color is then `20*log10(A / reference_amplitude)`. A tone at the reference is
0 dB; half that amplitude is about -6.02 dB. Raw PCM uses a reference of 32768;
unit-scale signals use 1. We do not normalize every image to its loudest point.

The default display clips colors to [-80, 0] dB. Silence maps to -80 dB without
computing log(0), and stronger-than-reference amplitudes reach the top color.
These are display limits; the stored complex coefficients remain unchanged.
Window correction retains the same leakage and overlap limitations as our
single-frame amplitude spectrum. The color values are not sound-pressure levels.

The STFT stores `(frames, bins)`, but the image requires `(frequency rows,
time columns)`, so plotting transposes the display data. Column m spans
`[times[m], times[m] + hop_size/sample_rate)`. Its analysis uses an entire
frame, so an overlapping frame can contain a later event than its start label.
In the tone-switch example, the column starting at 0.21875 s already sees part
of the switch at 0.25 s. This is expected, not early detection.

Frequency cell boundaries lie halfway between adjacent bins, with the outer
edges at DC and Nyquist. There is no interpolation between cells. In the example,
the final column ends at 0.46875 s although its window analyzes through 0.5 s.
This explicit convention also works for one frame or one frequency bin.

Rendering uses [Matplotlib's explicit-cell pcolormesh](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html).

## Onset strength from spectral flux

An onset is the beginning of a sound event. Before selecting onset times, we
compute a curve describing increases in spectral magnitude:

```text
M[m,k] = abs(STFT[m,k])
flux[0] = 0
flux[m] = sum_k(max(0, M[m,k] - M[m-1,k]))
```

The first frame has no predecessor, so its strength is zero even if it contains
a sound. Empty STFTs return an empty array. Magnitudes ignore phase, and clipping
each difference at zero makes decreases contribute nothing. For example, a
change from `[1, 4, 0]` to `[3, 1, 2]` gives `2 + 0 + 2 = 4`. Clipping must happen
before summing; otherwise a decrease could cancel an increase in another bin.

This baseline sums raw nonnegative-frequency FFT magnitudes, including DC and
Nyquist. There is no bin doubling, window-gain correction, log compression,
per-frame normalization, or smoothing. Multiplying audio by a constant gain
multiplies flux by its absolute value. Frame size and hop also affect the result;
the curve is not a probability and is not directly comparable across settings.

`flux[m]` is aligned with the current frame's start time, just like our STFT.
An event inside a window can affect a value labeled earlier than the event.
There is no universal time correction here: onset timestamp selection is
a separate step described below. A strong spectral change need not be a new note; noise,
frequency movement, and window boundary effects can also create increases.

Our known-answer tests include two isolated impulses. For a 64-sample Hann
window, each impulse contributes its windowed amplitude equally to all 33 FFT
bins. With a 16-sample hop and the chosen impulse alignment, its window weight
rises by 0.5 twice, producing two flux values of 16.5 for a unit impulse.
This tests both strength and frame alignment without a second FFT implementation.

The example uses damped 128 Hz bursts at 0.25, 0.75, and 1.25 seconds. The markers
shown as dashed lines are ground truth; crosses now mark selected peaks.
Its scalar reference checks the vectorized
calculation, while the tests establish known answers and behavior on silence,
steady tones, phase changes, and falling magnitudes.

Related reference: [librosa's onset-strength envelope](https://librosa.org/doc/0.10.2/generated/librosa.onset.onset_strength.html)
also uses positive spectral differences, but its defaults use log-power Mel
features and a mean aggregation. We do not claim numerical equivalence to those defaults.

## Peak picking and candidate onset times

The strength curve has one value per frame; a peak picker decides which values
are worth keeping. Our first picker uses an absolute threshold T and a minimum
time interval D. Candidate peaks must be strictly greater than T and greater
than the neighboring values on each side. A flat plateau counts once at its
earliest frame, provided both neighboring regions are lower. Peaks touching
the beginning or end of the curve are excluded because one side is unknown.

We process candidates from strongest to weakest. Keep a candidate only if it
is at least D seconds from every peak already kept. Ties favor the earlier
frame, and the final output is sorted by time. Equality at D is allowed, with
1e-12 relative tolerance for floating-point roundoff. This greedy offline rule
can choose one strong middle peak instead of two weaker outer peaks; it does
not maximize a global score and needs O(P²) work for P candidate peaks.

For `[0, 3, 0, 5, 0]`, both internal peaks qualify when T=2. If their times
are too close, the later height-5 peak suppresses the height-3 peak. Thresholds
must be adjusted for signal gain and STFT configuration because flux is raw.
There is no adaptive baseline, smoothing, or backtracking in this version.

With threshold 10 and minimum interval 0.1 s, our known bursts yield frame
indices 14, 46, 78, corresponding to 0.21875, 0.71875, 1.21875 s. Ground truth
is 0.25, 0.75, 1.25 s. Each signed error (candidate minus truth) is -0.03125 s.
Picking the correct flux maximum therefore does not guarantee precise onset
timing. No automatic window-center adjustment is made; refining the timing
requires tests with different transient shapes and positions within frames.

The example pairs each chronological candidate with the nearest unmatched
ground-truth event within two hops (31.25 ms). The fixture's widely spaced
events make that association unambiguous. Unmatched candidates count as false
positives; unmatched events count as misses. Report precision/recall together
with the tolerance and signed/absolute errors: passing within tolerance is not
zero error, and a three-burst fixture is not evidence of general music accuracy.

## Evaluating frame-center timestamps

A possible timing estimate is `frame_start + frame_size / (2*sample_rate)`.
This is the midpoint of the frame's time interval, not the midpoint between
its first and last sampled timestamps (which differs by half a sample).
We compare it with the existing start label using the same flux and peak rules.
The experiment checks that adding this constant does not change peak selection.

The fixed suite uses Fs=2048 Hz, frame lengths 64/128/256, and hops L/8, L/4,
and L/2. Each configuration places a single event at offsets 0, H/4, H/2,
and H-1 within the hop. There are three event shapes: a one-sample click, a
sharp damped 128 Hz burst, and a 128 Hz burst whose amplitude rises over 64
samples. The first nonzero sample is the known onset. All events have silent
padding before/after them; these are isolated events, not overlapping music.

Threshold 10 and minimum spacing 0.1 s are fixed before the sweep. We report
timing error only when exactly one candidate exists, without rejecting large
errors. Zero and multiple detections remain visible as separate counts. Scores
within 5/10/20/50 ms use all 108 event cases in their denominator, so misses
cannot disappear from the accuracy summary.

Initial mean absolute errors, conditional on one candidate:

| Signal | Paired cases | Frame start | Frame center |
| --- | --- | --- | --- |
| Click | 36/36 | 47.92 ms | 11.54 ms |
| Sharp burst | 36/36 | 43.36 ms | 7.07 ms |
| Slower attack | 28/36 | 27.64 ms | 16.24 ms |

Across the 100 paired cases, center labeling improves 84, worsens 12, and ties
4. All worsened cases have slower attacks. Eight slow-attack cases are missed
at the fixed threshold; there are no multiple-candidate cases. Eighteen silence
and steady-tone controls produce no candidates.

The shift fixes the original aligned sharp-burst example exactly, but that
does not make it a universal onset correction. With slower attacks, the largest
spectral increase may occur later in the envelope; shifting its frame label
forward can overshoot the beginning. We retain the existing public defaults
and compare the optional signal-based refinement below.

## Refining timestamps with local energy

The flux peak identifies a region where the spectrum changes. We can inspect
sample-level energy inside that region to estimate the start more precisely.
Our trailing mean-square curve is

`E[n] = (x[n]^2 + x[n-1]^2 + ... + x[n-W+1]^2) / W`.

Samples before zero are treated as zero. The curve has one value per input
sample, timestamped `n/Fs`. Squaring removes the sign of oscillations; averaging
smooths their variation. Unlike a centered window, a trailing window cannot
rise before the event. It can delay a threshold crossing, so there is no fixed
subtraction that makes every waveform exact. NumPy performs the direct
convolution; we implement the timing rule ourselves.

For each candidate, examine the closed interval `candidate ± search_radius`,
clipped to available sample timestamps. Define
`level = min(E) + fraction * (max(E) - min(E))` within that interval. Find the
first sample strictly above this level whose predecessor was at or below it,
up to the earliest maximum. If no crossing exists, retain the candidate time.
For example, energies `[2, 2, 4, 6, 10]` at fraction 0.25 give level 4 and a
crossing at index 3. A flat or already-falling region gives no crossing.

The evaluation fixes W=8 (3.90625 ms), fraction=0.1, and a radius of L/Fs around
the frame center. Ground truth is not an input to refinement. It reduces mean
absolute error from 11.25 ms to 3.14 ms on the same 100 detected events:
clicks and sharp bursts are exact, while slow attacks cross 11.23 ms late.
Compared with center timing, 84 improve, 7 worsen, and 9 tie. The eight missed
events stay missed. These are isolated events in silence, not music.

This estimates an energy crossing, which need not equal the first nonzero
sample or the perceived start. A weak preceding sound can attract the crossing;
noise can raise the local minimum. Separate candidates can refine to the same
crossing or change order, so results retain correspondence without claiming a
new spacing guarantee. Refinement is optional and does not change peak picking.

## Measuring volume and noise sensitivity

The sensitivity fixture is `y[n] = gain * event[n] + sigma * z[n]`, where `z`
is a seeded standard-normal sequence. Gain and noise amplitude vary independently;
we do not normalize the result. Sigma is the noise's theoretical standard
deviation, not the signal-to-noise ratio. The finite draw's RMS is also recorded.

Without added noise, scaling a signal by a positive gain scales its FFT
magnitudes and spectral flux by that gain. An absolute threshold of 10 therefore
has different consequences for quiet and loud events. Adding noise changes
magnitudes nonlinearly: random frame-to-frame increases can exceed the threshold.
In the saved sweep, gain-0.1 events in silence are all missed, while noise alone
at sigma=0.2 generates 6.67 false detections per second across three short controls.

For a single true event, we match the nearest candidate within 50 ms, count all
other candidates as false, and count an unmatched true event as missed. Refining
a timestamp can change this classification without changing candidate count.
Precision measures the fraction of candidates matched; recall measures the
fraction of true events matched. Both are needed: noise can generate many peaks,
some of which happen to fall inside the matching window. Timing-error means are
conditional on matches and cannot establish success by themselves.

The event gains, noise levels, and seeds are fixed before running the sweep.
Repeated seeds support paired comparisons but create correlated cases. Ground
truth is only used for scoring, never by the detector or refinement. These
measurements motivate testing a noise-aware threshold; they do not establish
which replacement will work or validate detection on music.

## Local adaptive onset threshold

An absolute threshold does not account for local background strength. Our first
optional alternative computes, for each frame i, the mean and population
standard deviation of flux in `[i-radius, i+radius]`, clipped at the boundaries:

`T[i] = max(floor, mean(local_flux) + multiplier * std(local_flux))`.

The peak picker requires `flux[i] > T[i]`. It keeps the existing local-maximum
and spacing rules. A plateau uses its first frame's threshold, even if a later
frame has a lower threshold. The helper returns the entire threshold curve so
we can inspect it separately from peak selection. Radius is measured in frames;
with hop 32 at 2048 Hz, radius 8 spans 125 ms on either side. The floor is in raw
flux units, as is the mean/std; the multiplier is dimensionless.

For `[1, 3, 5]`, radius 1 and multiplier 1, the first clipped neighborhood is
`[1, 3]`: mean 2, population std 1, giving threshold 3 if the floor is zero.
The center neighborhood's std is `sqrt(8/3)`. No padding or sample-variance
correction is applied. Centering uses future frames, so this is offline analysis.

With a zero floor, positive gain scales both flux and its threshold equally.
A nonzero floor breaks exact gain invariance but suppresses tiny numerical
changes. The test configuration uses floor 0.1, radius 8, multiplier 3. These
are experimental settings, not universal constants or a probabilistic
three-sigma guarantee: flux is nonnegative, correlated, and not assumed Gaussian.

The event contributes to its own local statistics. Broad or dense events can
raise the threshold and suppress themselves; clipped edge neighborhoods can
also be too small for a peak to exceed three standard deviations above the mean.
The evaluation confirms the tradeoff: on fresh noise seeds, false extras drop
from 603 to zero but matches drop from 360 to 262 out of 576. This motivates a
separate recall/false-detection study before selecting any default.

## Selecting parameters without using evaluation feedback

A threshold multiplier controls a tradeoff: stricter thresholds can suppress
false detections and real events together. We compare a small, declared grid
using tuning cases rather than selecting the setting with the nicest plot.

Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and their harmonic mean is
`F1 = 2*TP/(2*TP+FP+FN)`. False detections from noise-only controls contribute
to FP as well. This avoids choosing a detector that finds events by emitting
many unrelated peaks. F1 is a chosen balancing objective, not a universal DJ
quality measure; it depends on the case mix and matching tolerance.

The first search uses radius 8/12/16 and multiplier 1.5/2/2.5/3, floor 0.1,
with all other settings held fixed. Tuning seeds are 101/211/307. We select
maximum tuning F1, freeze the parameters, then evaluate on seeds 401/503/607.
The selected radius 12 means up to 187.5 ms on each side at hop 32 / Fs 2048;
its larger neighborhood and multiplier 2.5 let more isolated peaks survive
than the original radius 8 / multiplier 3 rule.

Selected held-out F1 is 0.849 versus 0.627 for the original adaptive setting
and 0.494 for fixed threshold 10. The selected method still misses 151/576
events. New seeds change noise draws, not event shapes; clean cases are shared.
These results support testing the selected setting on repeated-event sequences,
not claiming it generalizes to real music or changing production defaults.

## Future notes

For each new algorithm, document:

1. The problem it solves.
2. The mathematical model, units, and assumptions.
3. A small worked example and a simple implementation.
4. Tests using signals with known ground truth.
5. Comparison with an established implementation where useful.

Start with samples, sample rate, channels, PCM, amplitude, clipping,
quantization, Nyquist frequency, and aliasing. Then cover DFT, FFT, windowing,
and STFT as frequency analysis is implemented. Later topics include
autocorrelation, filters, similarity, graph optimization, and ML objectives.
