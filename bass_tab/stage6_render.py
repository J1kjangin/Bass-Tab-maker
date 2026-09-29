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
_DURATIONS = [(16, "1", False), (12, "2", True), (8, "2", False), (6, "4", True),
              (4, "4", False), (3, "8", True), (2, "8", False), (1, "16", False)]


_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_root(name: str) -> str:
    """"Am7" -> "A", "G#m7" -> "G#". The quality is dropped: the tab shows the root only."""
    return name[:2] if len(name) > 1 and name[1] in "#b" else name[:1]


def _note_name(midi: int) -> str:
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def _split(ticks: int) -> list[tuple[int, str, bool]]:
    """-> [(length in ticks, AlphaTex duration, dotted)] covering `ticks`."""
    out = []
    for size, dur, dotted in _DURATIONS:
        while ticks >= size:
            out.append((size, dur, dotted))
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

    # chord symbols: printed on the beat they start on, via the AlphaTex beat property {ch "X"}.
    # Only the root is printed (A, not Am7), so neighbouring bars sharing a root become one label.
    pending: list[tuple[int, str]] = []
    for tick, name in sorted(((c.tick, chord_root(c.name)) for c in tab.chords)):
        if not pending or pending[-1][1] != name:
            pending.append((tick, name))

    def effects(start: int, size: int, dotted: bool) -> str:
        """One brace group per beat: alphaTex rejects a second `{...}` on the same beat."""
        parts = ["d"] if dotted else []
        while pending and pending[0][0] < start:
            pending.pop(0)               # its beat has already gone by
        if pending and start <= pending[0][0] < start + size:
            parts.append('ch "%s"' % pending.pop(0)[1].replace('"', ""))
        return "{%s}" % " ".join(parts) if parts else ""

    def emit(start: int, end: int, first: str, rest: str) -> None:
        # place [start, end) splitting at bar lines; first piece uses `first`, the rest `rest`
        head = first
        while start < end:
            b = start // bar
            stop = min(end, (b + 1) * bar)
            for size, dur, dotted in _split(stop - start):
                bars[b].append(f"{head}.{dur}{effects(start, size, dotted)}")
                head = rest
                start += size

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
        f"\\tuning ({' '.join(_note_name(m) for m in tab.tuning)})",
        "\\clef F4",
        f"\\ts ({tab.beats_per_bar} 4)",
        f"\\tempo {round(tab.bpm)}",
    ]
    return "\n".join(header + [" |\n".join(" ".join(b) for b in bars)]) + "\n"
