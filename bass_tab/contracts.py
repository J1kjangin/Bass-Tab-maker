"""Data contracts between pipeline stages.

Every stage reads/writes files inside one job directory (jobs/{job_id}/).
This module is the single source of truth for those file names and shapes.
Stage owners must not change it; ask the orchestrator instead.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ---- job directory layout -------------------------------------------------
META = "meta.json"        # Stage 0 -> Meta
MIX_WAV = "mix.wav"       # Stage 0 -> 44.1 kHz, stereo, 16-bit PCM
BASS_WAV = "bass.wav"     # Stage 1 -> 44.1 kHz, stereo or mono, isolated bass
PITCH_NPZ = "pitch.npz"   # Stage 2 -> Pitch
BEATS_JSON = "beats.json" # Stage 3 -> Beats
NOTES_JSON = "notes.json" # Stage 4 -> list[Note]
TAB_JSON = "tab.json"     # Stage 5 -> Tab
TAB_TEX = "tab.alphatex"  # Stage 6 -> AlphaTex text

SR = 44100
PITCH_HOP_S = 0.01        # Stage 2 frame step (10 ms)
TICKS_PER_BEAT = 4        # 16th-note grid

# Standard 4-string bass. String numbers follow AlphaTex: 1 = G (highest) ... 4 = E (lowest).
OPEN_MIDI = {1: 43, 2: 38, 3: 33, 4: 28}
MAX_FRET = 20


class PipelineError(Exception):
    """Expected, user-facing failure (bad URL, live stream, no bass found, ...)."""


# ---- Stage 0 --------------------------------------------------------------
@dataclass
class Meta:
    source: str              # URL or original file path
    title: str
    duration: float          # seconds
    warnings: list[str] = field(default_factory=list)


# ---- Stage 2 --------------------------------------------------------------
@dataclass
class Pitch:
    time: np.ndarray         # (N,) seconds, frame centers
    f0: np.ndarray           # (N,) Hz; NaN where unvoiced
    confidence: np.ndarray   # (N,) 0..1 (torchcrepe periodicity)

    def save(self, path: Path) -> None:
        np.savez(path, time=self.time, f0=self.f0, confidence=self.confidence)

    @classmethod
    def load(cls, path: Path) -> "Pitch":
        d = np.load(path)
        return cls(d["time"], d["f0"], d["confidence"])


# ---- Stage 3 --------------------------------------------------------------
@dataclass
class Beats:
    beats: list[float]       # seconds, ascending
    downbeats: list[float]   # seconds, subset of beats
    bpm: float
    beats_per_bar: int = 4


# ---- Stage 4 --------------------------------------------------------------
@dataclass
class Note:
    midi: int
    start: float             # seconds (raw onset)
    end: float               # seconds (raw offset)
    tick: int                # quantized onset, 16th-note ticks from first downbeat
    length: int              # quantized duration in ticks, >= 1


# ---- Stage 5 --------------------------------------------------------------
@dataclass
class TabNote:
    midi: int
    tick: int
    length: int
    string: int              # 1..4, see OPEN_MIDI
    fret: int                # 0..MAX_FRET


@dataclass
class Tab:
    title: str
    bpm: float
    beats_per_bar: int
    notes: list[TabNote]


# ---- JSON helpers -----------------------------------------------------------
def save_json(obj, path: Path) -> None:
    if isinstance(obj, list):
        data = [asdict(o) for o in obj]
    else:
        data = asdict(obj)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_meta(path: Path) -> Meta:
    return Meta(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_beats(path: Path) -> Beats:
    return Beats(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_notes(path: Path) -> list[Note]:
    return [Note(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def load_tab(path: Path) -> Tab:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["notes"] = [TabNote(**n) for n in d["notes"]]
    return Tab(**d)
