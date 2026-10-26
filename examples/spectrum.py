"""Plot 220 Hz and 440 Hz tones with amplitudes 1 and 0.5."""

import math
from pathlib import Path

from autodj.audio import save_spectrum
from autodj.dsp import amplitude_spectrum


def main():
    sample_rate = 1760
    samples = [
        math.sin(2 * math.pi * 220 * n / sample_rate)
        + 0.5 * math.sin(2 * math.pi * 440 * n / sample_rate)
        for n in range(64)
    ]
    output = Path("outputs/spectra")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "two_tones.png"
    save_spectrum(samples, sample_rate, path, title="Two tones: 220 Hz + 440 Hz")
    frequencies, amplitudes = amplitude_spectrum(samples, sample_rate)
    for frequency, amplitude in zip(frequencies, amplitudes):
        if amplitude > 1e-10:
            print(f"{frequency:g} Hz: amplitude {amplitude:.3f}")
    print(path)


if __name__ == "__main__":
    main()
