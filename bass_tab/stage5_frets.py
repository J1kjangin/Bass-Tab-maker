"""Stage 5: choose (string, fret) for each note with a Viterbi/DP over positions.

Cost of moving from position a to b (weights from the design doc, tune by ear):
    FRET_WEIGHT * |fret_b - fret_a| + STRING_CHANGE if strings differ + OPEN_BONUS if fret_b == 0
The open bonus is also applied to the first note, so every note is scored the same way.
Negative costs are fine: the DP is a min-sum over a layered DAG (no cycles), so the
optimum is exact regardless of sign.

Out-of-range notes (no string/fret can play them) are octave-shifted into range
(below the lowest open string -> up, above the top string's MAX_FRET -> down). The returned
TabNote.midi is the shifted pitch, so tuning[string - 1] + fret == TabNote.midi always holds.
"""
from __future__ import annotations

from bass_tab.contracts import MAX_FRET, TUNINGS, Note, TabNote

FRET_WEIGHT = 2
STRING_CHANGE = 1
OPEN_BONUS = -2


def _in_range(midi: int, lowest: int, highest: int) -> int:
    while midi < lowest:
        midi += 12
    while midi > highest:
        midi -= 12
    return midi


def _candidates(midi: int, open_midi: dict[int, int]) -> list[tuple[int, int]]:
    return [(s, midi - o) for s, o in open_midi.items() if 0 <= midi - o <= MAX_FRET]


def _cost(prev: tuple[int, int] | None, cur: tuple[int, int]) -> int:
    c = OPEN_BONUS if cur[1] == 0 else 0
    if prev is not None:
        c += FRET_WEIGHT * abs(cur[1] - prev[1]) + (STRING_CHANGE if cur[0] != prev[0] else 0)
    return c


def assign(notes: list[Note], tuning=TUNINGS["standard"]) -> list[TabNote]:
    """tuning: open-string MIDI, string 1 (highest) first."""
    if not notes:
        return []
    open_midi = dict(enumerate(tuning, 1))
    lo, hi = min(tuning), max(tuning) + MAX_FRET
    pitches = [_in_range(n.midi, lo, hi) for n in notes]
    # layers[i]: candidate -> (total cost, best previous candidate)
    layers = [{c: (_cost(None, c), None) for c in _candidates(pitches[0], open_midi)}]
    for p in pitches[1:]:
        prev = layers[-1]
        layers.append({
            c: min(((pc + _cost(q, c), q) for q, (pc, _) in prev.items()), key=lambda t: t[0])
            for c in _candidates(p, open_midi)
        })
    pos = min(layers[-1], key=lambda c: layers[-1][c][0])
    path = [pos]
    for layer in reversed(layers[1:]):
        pos = layer[pos][1]
        path.append(pos)
    path.reverse()
    return [TabNote(midi=p, tick=n.tick, length=n.length, string=s, fret=f)
            for n, p, (s, f) in zip(notes, pitches, path)]
