"""Stage 6: Tab -> AlphaTex (alphaTab >= 1.7 syntax, no '.' section separator).

Syntax verified against https://www.alphatab.net/docs/alphatex/ (alphaTab 1.8.4):
- \\tuning lists strings high to low, string 1 first: (G2 D2 A1 E1).
- note = fret.string.duration; rest = r.duration; tie = -.string.duration; dot = {d}.
- \\ts (num den) and \\tempo are bar metadata placed before the first bar's beats.
Time signature is always beats_per_bar/4 (one beat = TICKS_PER_BEAT 16th ticks).
Monophonic: a note is cut at the next note's onset; notes sharing a tick keep the first.
"""
from __future__ import annotations

from bass_tab.contracts import TICKS_PER_BEAT, Tab

# 16th-note ticks -> AlphaTex duration (1 whole .. 16 sixteenth), dotted where needed.
# ponytail: greedy largest-first split, ignores beat grouping; add beat-aligned split if readability matters.
_DURATIONS = [(16, "1"), (12, "2{d}"), (8, "2"), (6, "4{d}"), (4, "4"), (3, "8{d}"), (2, "8"), (1, "16")]


def _split(ticks: int) -> list[str]:
    out = []
    for size, dur in _DURATIONS:
        while ticks >= size:
            out.append(dur)
            ticks -= size
    return out


def to_alphatex(tab: Tab) -> str:
    bar = tab.beats_per_bar * TICKS_PER_BEAT
    notes = sorted(tab.notes, key=lambda n: n.tick)
    # (start, end, fret, string) with monophonic clipping
    events = []
    for i, n in enumerate(notes):
        end = n.tick + n.length
        if i + 1 < len(notes):
            end = min(end, notes[i + 1].tick)
        if end > n.tick:
            events.append((n.tick, end, n.fret, n.string))
    total = max((e[1] for e in events), default=0)
    n_bars = max(1, -(-total // bar))
    bars: list[list[str]] = [[] for _ in range(n_bars)]

    def emit(start: int, end: int, first: str, rest: str) -> None:
        # place [start, end) splitting at bar lines; first piece uses `first`, the rest `rest`
        head = first
        while start < end:
            b = start // bar
            stop = min(end, (b + 1) * bar)
            for dur in _split(stop - start):
                bars[b].append(f"{head}.{dur}")
                head = rest
            start = stop

    cursor = 0
    for start, end, fret, string in events:
        if start > cursor:
            emit(cursor, start, "r", "r")
        emit(start, end, f"{fret}.{string}", f"-.{string}")
        cursor = end
    emit(cursor, n_bars * bar, "r", "r")

    title = tab.title.replace('"', "'")
    header = [
        f'\\title "{title}"',
        '\\track "Bass" { instrument 33 }',
        "\\staff { score tabs }",
        "\\tuning (G2 D2 A1 E1)",
        "\\clef F4",
        f"\\ts ({tab.beats_per_bar} 4)",
        f"\\tempo {round(tab.bpm)}",
    ]
    return "\n".join(header + [" |\n".join(" ".join(b) for b in bars)]) + "\n"
