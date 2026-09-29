import numpy as np
import soundfile as sf

from bass_tab.chords import bar_chroma, label_bars

SR = 22050


def _chord_audio(path, chords, bar_s=2.0):
    """Sine tones for each chord, one bar each, plus a weak click track."""
    out = []
    t = np.arange(int(bar_s * SR)) / SR
    for midis in chords:
        bar = sum(np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * t) for m in midis) / len(midis)
        env = np.minimum(1.0, np.exp(-t * 0.5))
        out.append((bar * env).astype(np.float32))
    y = np.concatenate(out)
    y[:: SR // 2] += 0.2          # clicks on every beat (120 BPM)
    sf.write(path, y, SR)
    return y


def test_labels_major_and_minor_bars(tmp_path):
    # C major (C E G), then A minor (A C E), then G7 (G B D F)
    path = tmp_path / "mix.wav"
    _chord_audio(path, [[60, 64, 67], [57, 60, 64], [55, 59, 62, 65]])
    beats = [i * 0.5 for i in range(13)]
    vecs, starts = bar_chroma(path, beats, 4)
    labels = label_bars(vecs)
    assert starts == [0.0, 2.0, 4.0]
    assert labels[0] == "C" and labels[1] == "Am" and labels[2] in ("G7", "G")
