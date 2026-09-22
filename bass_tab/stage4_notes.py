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


def _onset_frames(bass_wav: Path) -> np.ndarray:
    y, sr = librosa.load(bass_wav, sr=ONSET_SR, mono=True)
    t = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    return np.round(t / PITCH_HOP_S).astype(int)


def segment_frames(pitch: Pitch, onset_frames: np.ndarray) -> list[tuple[int, float, float]]:
    """-> [(midi, start_s, end_s)] before quantization."""
    voiced = ~np.isnan(pitch.f0)
    midi = np.full(len(pitch.f0), -1)
    midi[voiced] = np.round(librosa.hz_to_midi(pitch.f0[voiced])).astype(int)
    midi = _absorb_short(midi)

    n = len(midi)
    onsets = np.unique(np.clip(onset_frames, 0, n - 1))

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
    snap = lambda t: int(np.argmin(np.abs(grid - t)))  # noqa: E731

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
    raw = segment_frames(pitch, _onset_frames(job_dir / BASS_WAV))
    notes = quantize(raw, beats.beats, beats.downbeats, beats.beats_per_bar)
    save_json(notes, job_dir / NOTES_JSON)
    return notes
