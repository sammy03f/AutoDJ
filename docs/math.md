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
between bins spread across multiple bins (spectral leakage); we will study
windows later. This first implementation applies no window or zero padding.

`dft` uses two explicit loops: N output bins times N input samples, or O(N²)
work. FFT algorithms compute the same DFT in O(N log N) work. Our implementation
uses the standard library; NumPy provides the independent FFT comparison.
Known-answer tests check magnitude, phase, and normalization before comparison
with NumPy on real and complex signals, including odd lengths.

Reference convention: [NumPy's DFT documentation](https://numpy.org/doc/stable/reference/routines.fft.html).

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
