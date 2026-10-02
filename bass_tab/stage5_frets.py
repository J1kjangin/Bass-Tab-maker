"""Stage 5: choose (string, fret) for each note with a Viterbi/DP over positions.

Cost of moving from position a to b:
    FRET_WEIGHT * |fret_b - fret_a| + STRING_CHANGE if strings differ
  + STRING_PREF per string above the lowest + HIGH_FRET_COST per fret above HIGH_FRET_FROM
The two position terms are also applied to the first note, so every note is scored the same way.
The design doc's open-string bonus is gone: on the four reference songs it only pulled notes
onto an open string the player did not use (103 -> 102 of 131 fingerings).

Out-of-range notes (no string/fret can play them) are octave-shifted into range
(below the lowest open string -> up, above the top string's MAX_FRET -> down). The returned
TabNote.midi is the shifted pitch, so tuning[string - 1] + fret == TabNote.midi always holds.
"""
from __future__ import annotations

from bass_tab.contracts import MAX_FRET, TUNINGS, Note, TabNote  # noqa: F401  (TUNINGS re-used)

FRET_WEIGHT = 2
STRING_CHANGE = 1
# Players hold a position and reach for a higher fret on a lower string rather than dropping to
# the next string up; without this the cost preferred the opposite (same pitch, 5 frets lower,
# one string up). This penalises the thinner strings. 0 = off.
STRING_PREF = 0.25
# The string preference alone is unbounded: it moved an E2-G#2 run to fret 12-16 of the E string
# because that is the lowest string at any height. Frets above HIGH_FRET_FROM cost extra, which
# keeps the preference to the 5-7 fret trade a player actually makes. Measured: with FROM=9 every
# cost in 0.5-2.0 and every STRING_PREF in 0.15-0.35 gives the same 103/131 fingerings, so both
# sit in the middle of a wide plateau.
HIGH_FRET_FROM = 9
HIGH_FRET_COST = 1.0
# Auto tuning: a lower tuning is only chosen when notes it alone can play really sound.
# Reference songs: the Drop D one had 76 such notes / 16.9 s, the three standard ones 1-2 / 0.1 s.
LOW_MIN_NOTES = 5
LOW_MIN_SECONDS = 1.0


def detect_tuning(notes: list[Note], tunings=TUNINGS) -> str:
    """Pick the tuning name from what the bass actually plays: standard unless notes below its
    lowest string sound long enough to be real. Only the lowest string differs between the
    supported tunings, so this is a single threshold, not a search."""
    best = "standard"
    for name, tuning in sorted(tunings.items(), key=lambda kv: -min(kv[1])):
        floor = min(tuning)
        low = [n for n in notes if floor <= n.midi < min(tunings[best])]
        if len(low) >= LOW_MIN_NOTES and sum(n.end - n.start for n in low) >= LOW_MIN_SECONDS:
            best = name
    return best


def _in_range(midi: int, lowest: int, highest: int) -> int:
    while midi < lowest:
        midi += 12
    while midi > highest:
        midi -= 12
    return midi


def _candidates(midi: int, open_midi: dict[int, int]) -> list[tuple[int, int]]:
    return [(s, midi - o) for s, o in open_midi.items() if 0 <= midi - o <= MAX_FRET]


def _cost(prev: tuple[int, int] | None, cur: tuple[int, int], n_strings: int = 4) -> float:
    c = STRING_PREF * (n_strings - cur[0])      # string 1 (thinnest) costs the most
    c += HIGH_FRET_COST * max(0, cur[1] - HIGH_FRET_FROM)
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
    ns = len(tuning)
    layers = [{c: (_cost(None, c, ns), None) for c in _candidates(pitches[0], open_midi)}]
    for p in pitches[1:]:
        prev = layers[-1]
        layers.append({
            c: min(((pc + _cost(q, c, ns), q) for q, (pc, _) in prev.items()), key=lambda t: t[0])
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
