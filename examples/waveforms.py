"""Save small synthetic waveform examples for visual inspection."""

import math
from pathlib import Path

from autodj.audio import WavAudio, apply_gain, save_waveform


def main():
    output = Path("outputs/waveforms")
    output.mkdir(parents=True, exist_ok=True)
    rate = 8000
    # Twenty milliseconds shows individual oscillations clearly.
    original = WavAudio(rate, 1, tuple(
        (round(20000 * math.sin(2 * math.pi * 220 * n / rate)),)
        for n in range(160)
    ))
    for name, gain in (("original", 1), ("quieter", 0.5), ("clipped", 2)):
        path = output / f"{name}.png"
        save_waveform(apply_gain(original, gain), path,
                      title=f"220 Hz tone — {name} (gain {gain:g})")
        print(path)
    stereo = WavAudio(rate, 2, tuple(
        (frame[0], round(10000 * math.sin(2 * math.pi * 440 * n / rate)))
        for n, frame in enumerate(original.samples)
    ))
    save_waveform(stereo, output / "stereo.png", title="Stereo — 220 Hz left / 440 Hz right")
    print(output / "stereo.png")


if __name__ == "__main__":
    main()
