"""Stage 4: frame pitch + beats + bass onsets -> quantized notes (notes.json).

Onset-driven: each bass onset (pluck) starts a note, which fixes the design doc's
pitch-only rule merging repeated plucks of one pitch. Extra boundaries without an onset:
a voiced run starting > ONSET_LAG_FRAMES after the last onset (soft attack), or a pitch
change held >= LEGATO_MIN_FRAMES (hammer-on/slide). A note's pitch is the most common
frame pitch inside its interval, its end the last voiced frame before the next boundary.

Measured on the real test song: torchcrepe turns voiced 20-60 ms after the backtracked
onset, so pitch-change boundaries alone split one pluck into several short fragments.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .contracts import (BASS_WAV, BEATS_JSON, NOTES_JSON, PITCH_HOP_S, PITCH_NPZ,
                        TICKS_PER_BEAT, Note, Pitch, load_beats, save_json)

MIN_RUN_FRAMES = 3        # 30 ms: shorter pitch excursions are treated as jitter
MIN_NOTE_S = 0.05         # design doc; voiced time required inside a note interval
ONSET_SR = 22050
ONSET_LAG_FRAMES = 12     # voicing within 120 ms after an onset belongs to that onset
LEGATO_MIN_FRAMES = 10    # a pitch change without onset must hold 100 ms to count
GAP_FILL_FRAMES = 5       # unvoiced dropouts this short inside one pitch are bridged
ONSET_BACKTRACK = True    # librosa backtrack: move onsets to the preceding energy minimum
# Snapping cost = distance in 16ths + penalty: an ambiguous onset (players drift +-50 ms,
# half a 16th at 130 BPM is 58 ms) prefers the beat, then the 8th, over an odd 16th.
# Reference intro (41 notes): exact matches 32 -> 37; plateau from (0.1, 0.35) up to (0.2, 0.5).
# Smoothing the beat grid against beat_this's 20 ms jitter was tried and did not help.
METER_PRIOR = (0.1, 0.35)  # (8th, 16th) penalties
# Reference intro: real same-pitch re-plucks rose +15/+32 dB, spurious onsets -0.6..+1 dB.
REPLUCK_DB = 6.0


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end_exclusive, label) runs; label -1 = unvoiced."""
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def _absorb_short(labels: np.ndarray) -> np.ndarray:
    """Relabel voiced runs shorter than MIN_RUN_FRAMES to the preceding voiced label."""
    out = labels.copy()
    for s, e, lab in _runs(labels):
        if lab >= 0 and e - s < MIN_RUN_FRAMES and s > 0 and out[s - 1] >= 0:
            out[s:e] = out[s - 1]
    return out


def _fill_gaps(labels: np.ndarray) -> np.ndarray:
    """Bridge short unvoiced dropouts between two frames of the same pitch."""
    out = labels.copy()
    runs = _runs(labels)
    for (_, _, a), (s, e, lab), (_, _, b) in zip(runs, runs[1:], runs[2:]):
        if lab < 0 and a >= 0 and a == b and e - s <= GAP_FILL_FRAMES:
            out[s:e] = a
    return out


def _onset_frames(bass_wav: Path) -> np.ndarray:
    y, sr = librosa.load(bass_wav, sr=ONSET_SR, mono=True)
    t = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=ONSET_BACKTRACK)
    return np.round(t / PITCH_HOP_S).astype(int)


def _frame_rms(bass_wav: Path, n: int) -> np.ndarray:
    y, sr = librosa.load(bass_wav, sr=16000, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=160, center=True)[0]
    return np.pad(rms, (0, max(0, n - len(rms))))[:n]


def _is_repluck(k: int, midi: np.ndarray, rms: np.ndarray) -> bool:
    """An onset inside a note of unchanged pitch counts only if the level jumps."""
    n = len(midi)
    before, after = midi[max(k - 1, 0)], midi[min(k + 3, n - 1)]
    if k < 4 or k + 6 > n or before < 0 or before != after:
        return True  # pitch change, attack from silence, or edge of the song: keep
    rise = rms[k + 2:k + 6].max() / max(rms[k - 4:k].min(), 1e-9)
    return 20 * np.log10(max(rise, 1e-9)) >= REPLUCK_DB


def segment_frames(pitch: Pitch, onset_frames: np.ndarray,
                   rms: np.ndarray | None = None) -> list[tuple[int, float, float]]:
    """-> [(midi, start_s, end_s)] before quantization. rms: per-frame level for the
    same-pitch re-pluck check (skipped when None)."""
    voiced = ~np.isnan(pitch.f0)
    midi = np.full(len(pitch.f0), -1)
    midi[voiced] = np.round(librosa.hz_to_midi(pitch.f0[voiced])).astype(int)
    midi = _absorb_short(_fill_gaps(midi))

    n = len(midi)
    onsets = np.unique(np.clip(onset_frames, 0, n - 1))
    if rms is not None:
        onsets = np.array([k for k in onsets if _is_repluck(int(k), midi, rms)], dtype=int)

    def after_onset(k: int) -> bool:
        i = np.searchsorted(onsets, k, side="right") - 1
        return i >= 0 and k - onsets[i] <= ONSET_LAG_FRAMES

    bounds = set(onsets.tolist())
    for s, e, lab in _runs(midi):
        if lab < 0 or after_onset(s):
            continue
        if s == 0 or midi[s - 1] < 0 or e - s >= LEGATO_MIN_FRAMES:
            bounds.add(s)  # soft attack after silence, or held pitch change (legato)
    edges = sorted(bounds) + [n]

    notes = []
    for a, b in zip(edges, edges[1:]):
        v = np.flatnonzero(midi[a:b] >= 0)
        if len(v) * PITCH_HOP_S < MIN_NOTE_S:
            continue
        lab = int(np.bincount(midi[a:b][v]).argmax())
        notes.append((lab, float(pitch.time[a]), float(pitch.time[a + v[-1]] + PITCH_HOP_S)))
    return notes


def _grid(beats: np.ndarray, t_min: float, t_max: float) -> tuple[np.ndarray, int]:
    """16th-note grid following the beat times, extended by the median period on both ends.
    Returns (grid, number of beats prepended)."""
    period = float(np.median(np.diff(beats)))
    pre = max(0, int(np.ceil((beats[0] - t_min) / period)))
    post = max(0, int(np.ceil((t_max - beats[-1]) / period))) + 1
    full = np.r_[beats[0] - period * np.arange(pre, 0, -1), beats,
                 beats[-1] + period * np.arange(1, post + 1)]
    grid = np.concatenate([np.linspace(a, b, TICKS_PER_BEAT, endpoint=False)
                           for a, b in zip(full[:-1], full[1:])] + [full[-1:]])
    return grid, pre


def quantize(raw: list[tuple[int, float, float]], beats: list[float], downbeats: list[float],
             beats_per_bar: int) -> list[Note]:
    if not raw:
        return []
    b = np.asarray(beats, dtype=float)
    grid, pre = _grid(b, raw[0][1], raw[-1][2])
    step = np.diff(grid, append=grid[-1] + (grid[-1] - grid[-2]))
    # grid index % 4: 0 = beat, 2 = 8th, 1/3 = 16th
    penalty = np.array([0.0, METER_PRIOR[1], METER_PRIOR[0], METER_PRIOR[1]])

    def snap(t: float) -> int:
        k = int(np.argmin(np.abs(grid - t)))
        cand = [i for i in (k - 1, k, k + 1) if 0 <= i < len(grid)]
        return min(cand, key=lambda i: abs(grid[i] - t) / step[min(i, k)] + penalty[i % 4])

    # tick 0 = first downbeat; step back whole bars so no note lands before tick 0.
    # ponytail: one global bar phase; misdetected downbeats later in the song are ignored.
    first_db = downbeats[0] if downbeats else beats[0]
    origin = (pre + int(np.argmin(np.abs(b - first_db)))) * TICKS_PER_BEAT
    bar = beats_per_bar * TICKS_PER_BEAT
    first = snap(raw[0][1])
    while origin > first:
        origin -= bar

    by_tick: dict[int, Note] = {}
    for midi, start, end in raw:
        s, e = snap(start), snap(end)
        n = Note(midi=midi, start=start, end=end, tick=s - origin, length=max(1, e - s))
        old = by_tick.get(n.tick)
        if old is None or (n.end - n.start) > (old.end - old.start):
            by_tick[n.tick] = n  # two notes on one 16th: keep the longer one
    return [by_tick[k] for k in sorted(by_tick)]


def segment(job_dir: Path) -> list[Note]:
    job_dir = Path(job_dir)
    pitch = Pitch.load(job_dir / PITCH_NPZ)
    beats = load_beats(job_dir / BEATS_JSON)
    bass = job_dir / BASS_WAV
    raw = segment_frames(pitch, _onset_frames(bass), _frame_rms(bass, len(pitch.f0)))
    notes = quantize(raw, beats.beats, beats.downbeats, beats.beats_per_bar)
    save_json(notes, job_dir / NOTES_JSON)
    return notes
