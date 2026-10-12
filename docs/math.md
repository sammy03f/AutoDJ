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
