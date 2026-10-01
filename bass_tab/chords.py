"""Chord symbols above the tab: chroma of the original mix matched against chord templates.

mix.wav + beats.json -> chords.json. One label per SEGMENT_BEATS beats (half a 4/4 bar), then
runs of the same label are merged, so the tab shows a symbol only where the chord changes.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .contracts import BEATS_JSON, CHORDS_JSON, MIX_WAV, Chord, load_beats, save_json

SR = 22050
HOP = 512
HARMONIC_MARGIN = 3.0     # librosa HPSS margin: keep the pitched part, drop drums
# Beats per analysed segment. 2 (half a 4/4 bar) beat a whole bar on the 19 reference bars,
# 13/19 -> 18/19: songs change chord mid-bar, and a bar-long window also smears across a wrong
# bar phase. 1 beat (17/19) and 3 beats (13/19) are both worse.
SEGMENT_BEATS = 2
SELF_BONUS = 0.06         # score added for keeping the previous bar's chord (chords are held)
MIN_SCORE = 0.55          # below this the bar is left unlabelled (intro noise, silence)

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITIES = {  # printed suffix -> semitones above the root
    "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
}
# sus4 (0,5,7) and dim (0,3,6) were dropped: on the reference song they swallowed ambiguous bars
# (F#m7 read as Bsus4). Without them every root of that intro is right (7/7, full name 5/7).


def _templates() -> tuple[np.ndarray, list[str]]:
    rows, labels = [], []
    for suffix, intervals in QUALITIES.items():
        for root in range(12):
            t = np.zeros(12)
            for i in intervals:
                t[(root + i) % 12] = 1
            rows.append(t / np.linalg.norm(t))
            labels.append(NAMES[root] + suffix)
    return np.array(rows), labels


TEMPLATES, LABELS = _templates()


def bar_chroma(mix: Path, beats: list[float], beats_per_segment: int) -> tuple[np.ndarray, list[float]]:
    """Mean chroma per segment (unit length) and each segment's start time."""
    y, _ = librosa.load(mix, sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y, margin=HARMONIC_MARGIN),
                                        sr=SR, hop_length=HOP)
    frames = librosa.time_to_frames(np.asarray(beats), sr=SR, hop_length=HOP)
    vecs, starts = [], []
    for i in range(0, len(frames) - beats_per_segment, beats_per_segment):
        a, b = int(frames[i]), int(frames[i + beats_per_segment])
        if b <= a:
            continue
        v = chroma[:, a:b].mean(axis=1)
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
        starts.append(float(beats[i]))
    return np.array(vecs) if vecs else np.zeros((0, 12)), starts


def label_bars(vecs: np.ndarray) -> list[str | None]:
    """Best template per segment, with a bonus for repeating the previous segment's chord."""
    out: list[str | None] = []
    prev = None
    for v in vecs:
        scores = TEMPLATES @ v
        if prev is not None:
            scores = scores + SELF_BONUS * (np.array(LABELS) == prev)
        k = int(np.argmax(scores))
        if scores[k] < MIN_SCORE:
            out.append(None)
            continue
        prev = LABELS[k]
        out.append(prev)
    return out


def detect(job_dir: Path) -> list[Chord]:
    job_dir = Path(job_dir)
    beats = load_beats(job_dir / BEATS_JSON)
    vecs, starts = bar_chroma(job_dir / MIX_WAV, beats.beats, SEGMENT_BEATS)
    chords: list[Chord] = []
    for name, start in zip(label_bars(vecs), starts):
        if name and (not chords or chords[-1].name != name):
            chords.append(Chord(name=name, start=start))
    save_json(chords, job_dir / CHORDS_JSON)
    return chords
