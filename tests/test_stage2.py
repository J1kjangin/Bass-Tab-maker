import numpy as np
import soundfile as sf

from bass_tab.contracts import PITCH_NPZ, SR, Pitch
from bass_tab.stage2_pitch import track

NOTES = [28, 33, 38, 43, 45, 48]  # E1 A1 D2 G2 A2 C3
NOTE_S, GAP_S = 0.4, 0.2


def _bass_line(rng):
    """Harmonic-rich plucked tones separated by near-silent (-80 dB noise) gaps."""
    parts, spans, t0 = [], [], 0.0
    gap = lambda: 1e-4 * rng.standard_normal(int(GAP_S * SR))
    for m in NOTES:
        parts.append(gap())
        t0 += GAP_S
        f = 440 * 2 ** ((m - 69) / 12)
        t = np.arange(int(NOTE_S * SR)) / SR
        tone = sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 6))
        env = np.minimum(1, t / 0.005) * np.minimum(1, (NOTE_S - t) / 0.01)
        parts.append(0.3 * tone * env * np.exp(-2 * t))
        spans.append((t0, t0 + NOTE_S))
        t0 += NOTE_S
    parts.append(gap())
    return np.concatenate(parts).astype(np.float32), spans


def test_track_synthetic_bass_line(tmp_path):
    audio, spans = _bass_line(np.random.default_rng(0))
    sf.write(tmp_path / "bass.wav", np.stack([audio, audio], 1), SR)  # stereo input

    p = track(tmp_path)

    assert np.allclose(np.diff(p.time), 0.01)
    assert len(p.time) == len(p.f0) == len(p.confidence)
    saved = Pitch.load(tmp_path / PITCH_NPZ)
    assert np.array_equal(saved.f0, p.f0, equal_nan=True)

    for m, (a, b) in zip(NOTES, spans):
        mid = (p.time > a + 0.1) & (p.time < b - 0.1)
        f = p.f0[mid]
        assert np.isfinite(f).mean() > 0.9, m
        assert round(np.median(librosa_midi(f[np.isfinite(f)]))) == m

    # Middle of each gap (and the lead-in) must be unvoiced.
    gap_mids = [spans[0][0] - GAP_S / 2] + [a - GAP_S / 2 for a, _ in spans[1:]]
    for g in gap_mids:
        near = np.abs(p.time - g) < 0.04
        assert np.isnan(p.f0[near]).all(), g


def librosa_midi(f):
    return 69 + 12 * np.log2(f / 440)
