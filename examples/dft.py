"""Find two known tones in a short signal and compare with NumPy's FFT."""

import math

import numpy as np

from autodj.dsp import dft


def main():
    sample_rate = 1760
    size = 64
    samples = [
        math.sin(2 * math.pi * 220 * n / sample_rate)
        + 0.5 * math.sin(2 * math.pi * 440 * n / sample_rate)
        for n in range(size)
    ]
    spectrum = dft(samples)
    reference = np.fft.fft(samples)
    print(f"{size} samples, {sample_rate / size:g} Hz per bin")
    print("Positive-frequency peaks (bin-centered tones):")
    for k in range(1, size // 2):
        # For a real, bin-centered sinusoid, amplitude = 2*|X[k]|/N.
        amplitude = 2 * abs(spectrum[k]) / size
        if amplitude > 1e-10:
            print(f"  {k * sample_rate / size:g} Hz: amplitude {amplitude:.3f}")
    error = max(abs(a - b) for a, b in zip(spectrum, reference))
    print(f"Maximum difference from NumPy FFT: {error:.3e}")


if __name__ == "__main__":
    main()
