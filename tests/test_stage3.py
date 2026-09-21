import numpy as np
import pytest
import soundfile as sf

from bass_tab.contracts import SR, PipelineError, load_beats
from bass_tab.stage3_beats import track

DUR = 20.0


def _drums(bpm: float, path) -> np.ndarray:
    """4/4 kick/snare/hat pattern with a crash + loud kick on each bar start. Returns true beat times."""
    rng = np.random.default_rng(0)
    n = int(DUR * SR)
    x = np.zeros(n)
    t = np.arange(int(0.4 * SR)) / SR

    def add(start, sig):
        i = int(start * SR)
        seg = sig[: n - i]
        x[i:i + len(seg)] += seg

    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
    snare = rng.standard_normal(len(t)) * np.exp(-t * 20) * 0.5
    hat = rng.standard_normal(len(t)) * np.exp(-t * 80) * 0.15
    crash = rng.standard_normal(len(t)) * np.exp(-t * 4) * 0.3

    beat = 60.0 / bpm
    beats = np.arange(0.1, DUR - 0.5, beat)
    for k, b in enumerate(beats):
        if k % 4 == 0:
            add(b, 1.5 * kick + crash)
        elif k % 4 == 2:
            add(b, kick)
        else:
            add(b, snare)
        add(b, hat)
        add(b + beat / 2, hat)
    x /= np.abs(x).max() * 1.1
    sf.write(path, np.stack([x, x], axis=1), SR, subtype="PCM_16")
    return beats


@pytest.mark.parametrize("bpm", [100, 128])
def test_track_synthetic(tmp_path, bpm):
    truth = _drums(bpm, tmp_path / "mix.wav")
    res = track(tmp_path)

    assert abs(res.bpm - bpm) <= 2
    err = np.abs(truth[:, None] - np.array(res.beats)[None, :]).min(axis=1)
    assert np.mean(err <= 0.040) >= 0.9
    assert set(res.downbeats) <= set(res.beats)
    assert load_beats(tmp_path / "beats.json") == res

    # report only: downbeat accuracy on synthetic audio
    true_db = truth[::4]
    hit = np.mean([np.abs(np.array(res.downbeats) - d).min() <= 0.040 for d in true_db]) if res.downbeats else 0.0
    print(f"\n{bpm} BPM: est {res.bpm:.2f}, beats_per_bar={res.beats_per_bar}, "
          f"{len(res.beats)} beats, {len(res.downbeats)} downbeats, true-downbeat hit rate {hit:.0%}")


def test_silence_raises(tmp_path):
    sf.write(tmp_path / "mix.wav", np.zeros((SR * 5, 2)), SR, subtype="PCM_16")
    with pytest.raises(PipelineError):
        track(tmp_path)
