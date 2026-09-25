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
