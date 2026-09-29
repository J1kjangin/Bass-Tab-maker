import librosa
import numpy as np
import soundfile as sf

from bass_tab.contracts import Beats, Pitch, load_notes, save_json
from bass_tab.stage4_notes import quantize, segment, segment_frames

SR = 44100
# (midi, hz, start_s, dur_s); second A1 is a re-pluck of the same pitch with no f0 gap
PHRASE = [(33, 55.0, 0.5, 0.5), (33, 55.0, 1.0, 0.5), (38, 73.42, 1.5, 0.25), (28, 41.2, 2.0, 1.0)]


def _write_job(d):
    n = int(3.5 * SR)
    y = np.zeros(n, np.float32)
    for _, hz, start, dur in PHRASE:
        t = np.arange(int(dur * SR)) / SR
        i = int(start * SR)
        y[i:i + len(t)] = 0.5 * (2 * ((hz * t) % 1) - 1) * np.exp(-t * 4)  # plucked decay
    sf.write(d / "bass.wav", np.stack([y, y], 1), SR)

    time = np.arange(350) * 0.01
    f0 = np.full(350, np.nan)
    for _, hz, start, dur in PHRASE:
        f0[int(round(start * 100)):int(round((start + dur) * 100))] = hz
    f0[160] = 77.78  # 1-frame semitone wobble inside D2 must not split it
    Pitch(time, f0, np.where(np.isnan(f0), 0.1, 0.9)).save(d / "pitch.npz")

    beats = [0.5 + 0.5 * k for k in range(7)]
    save_json(Beats(beats, beats[::4], 120.0, 4), d / "beats.json")


def test_segment_splits_repluck_and_quantizes(tmp_path):
    _write_job(tmp_path)
    notes = segment(tmp_path)
    got = [(n.midi, n.tick, n.length) for n in notes]
    assert got == [(33, 0, 4), (33, 4, 4), (38, 8, 2), (28, 12, 8)], got
    assert load_notes(tmp_path / "notes.json") == notes


def test_same_pitch_onset_needs_level_jump():
    f0 = np.full(100, 55.0)
    p = Pitch(np.arange(100) * 0.01, f0, np.full(100, 0.9))
    flat = np.full(100, 0.1)
    jump = flat.copy(); jump[52:] = 0.4                 # +12 dB re-pluck at frame 50
    onsets = np.array([0, 50])
    assert len(segment_frames(p, onsets, flat)) == 1    # spurious onset ignored
    assert len(segment_frames(p, onsets, jump)) == 2    # real re-pluck kept


def test_pickup_note_shifts_origin_back_a_bar():
    beats = [1.0 + 0.5 * k for k in range(8)]           # 120 BPM, first downbeat at 1.0 s
    notes = quantize([(40, 0.75, 1.0), (43, 1.0, 1.25)], beats, [1.0, 3.0], 4)
    # pickup 0.25 s before the downbeat -> 2 ticks before bar 2 (origin moved back 16 ticks)
    assert [(n.tick, n.length) for n in notes] == [(14, 2), (16, 2)]


def _frames(runs):
    """runs: [(midi or None, n_frames)] -> Pitch with 10 ms frames."""
    f0 = []
    for midi, n in runs:
        f0 += [np.nan if midi is None else float(librosa.midi_to_hz(midi))] * n
    f0 = np.array(f0)
    return Pitch(np.arange(len(f0)) * 0.01, f0, np.where(np.isnan(f0), 0.1, 0.9))


def test_unplucked_glide_is_absorbed_but_plucked_run_is_kept():
    p = _frames([(33, 40), (34, 8), (36, 40)])       # A1, 80 ms of A#1, C2
    flat = np.full(88, 0.1)
    glide = segment_frames(p, np.array([0, 48]), flat, tick_s=0.115)
    assert [n[0] for n in glide] == [33, 36]         # the 80 ms step was not plucked
    picked = segment_frames(p, np.array([0, 40, 48]), flat, tick_s=0.115)
    assert [n[0] for n in picked] == [33, 34, 36]    # same audio, but that step was plucked


def test_unvoiced_lead_in_moves_the_note_start():
    # ghost click at the onset, pitch only 80 ms later (still inside ONSET_LAG_FRAMES, so the
    # note keeps that onset): the start moves to the pitch, minus LEAD_TRIM_BACK
    p = _frames([(None, 8), (33, 40)])
    notes = segment_frames(p, np.array([0]), np.full(48, 0.1), tick_s=0.115)
    assert len(notes) == 1 and abs(notes[0][1] - 0.06) < 1e-9



def test_unplucked_same_pitch_continues_the_note():
    # pitch drops out for 100 ms and comes back on the same note without a pluck
    p = _frames([(33, 30), (None, 10), (33, 30)])
    notes = segment_frames(p, np.array([0]), np.full(70, 0.1), tick_s=0.115)
    assert [n[0] for n in notes] == [33] and abs(notes[0][2] - 0.70) < 1e-9


def test_grid_shift_keeps_a_slightly_late_band_on_the_beat():
    beats = [1.0 + 0.5 * k for k in range(6)]          # 120 BPM, 16th = 0.125 s
    lag = 0.05                                          # 0.4 of a 16th: within half a tick
    late = [(33, 1.0 + lag, 1.4), (35, 1.5 + lag, 1.9), (36, 2.0 + lag, 2.4)]
    assert [n.tick for n in quantize(late, beats, [1.0, 3.0], 4)] == [0, 4, 8]
