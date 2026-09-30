"""Tab -> MusicXML 4.0 (score-partwise), a single 4-line TAB staff with fret numbers.

AlphaTab can read MusicXML but not write it, so this is written here. Durations use
divisions=4 per quarter note, i.e. one division per 16th tick, matching contracts.TICKS_PER_BEAT.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

from .contracts import TICKS_PER_BEAT, Tab
from .stage6_render import _split, chord_root

DIVISIONS = TICKS_PER_BEAT            # per quarter note
_STEPS = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
_ALTER = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
_TYPES = {"1": "whole", "2": "half", "4": "quarter", "8": "eighth", "16": "16th"}


def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrib) -> ET.Element:
    el = ET.SubElement(parent, tag, {k: str(v) for k, v in attrib.items()})
    if text is not None:
        el.text = text
    return el


def _attributes(measure: ET.Element, tab: Tab) -> None:
    at = _sub(measure, "attributes")
    _sub(at, "divisions", str(DIVISIONS))
    _sub(_sub(at, "key"), "fifths", "0")
    time = _sub(at, "time")
    _sub(time, "beats", str(tab.beats_per_bar))
    _sub(time, "beat-type", "4")
    clef = _sub(at, "clef")
    _sub(clef, "sign", "TAB")
    _sub(clef, "line", "5")
    details = _sub(at, "staff-details")
    _sub(details, "staff-lines", str(len(tab.tuning)))
    for i, midi in enumerate(tab.tuning):          # tuning[0] = string 1 = highest = top line
        st = _sub(details, "staff-tuning", line=len(tab.tuning) - i)
        _sub(st, "tuning-step", _STEPS[midi % 12])
        if _ALTER[midi % 12]:
            _sub(st, "tuning-alter", "1")
        _sub(st, "tuning-octave", str(midi // 12 - 1))


def _tempo(measure: ET.Element, bpm: float) -> None:
    direction = _sub(measure, "direction", placement="above")
    metronome = _sub(_sub(direction, "direction-type"), "metronome")
    _sub(metronome, "beat-unit", "quarter")
    _sub(metronome, "per-minute", str(round(bpm)))
    _sub(direction, "sound", tempo=str(round(bpm)))


def _harmony(measure: ET.Element, name: str) -> None:
    """Root only, like the tab: <kind text=""> keeps readers from printing a quality."""
    harmony = _sub(measure, "harmony")
    root = _sub(harmony, "root")
    _sub(root, "root-step", name[0])
    if len(name) > 1:
        _sub(root, "root-alter", "1" if name[1] == "#" else "-1")
    kind = _sub(harmony, "kind", "none")
    kind.set("text", "")          # "text" is a parameter name of _sub, so set it directly


def _note(measure: ET.Element, ticks: int, type_: str, dotted: bool,
          pos: tuple[int, int, int] | None, tie: str | None) -> None:
    """pos = (midi, string, fret); None writes a rest. tie = None | 'start' | 'stop' | 'both'."""
    note = _sub(measure, "note")
    if pos is None:
        _sub(note, "rest")
    else:
        midi, _, _ = pos
        pitch = _sub(note, "pitch")
        _sub(pitch, "step", _STEPS[midi % 12])
        if _ALTER[midi % 12]:
            _sub(pitch, "alter", "1")
        _sub(pitch, "octave", str(midi // 12 - 1))
    _sub(note, "duration", str(ticks))
    for kind in ("stop", "start"):                  # MusicXML order: stop before start
        if tie in (kind, "both"):
            _sub(note, "tie", type=kind)
    _sub(note, "voice", "1")
    _sub(note, "type", type_)
    if dotted:
        _sub(note, "dot")
    if pos is None and tie is None:
        return
    notations = _sub(note, "notations")
    for kind in ("stop", "start"):
        if tie in (kind, "both"):
            _sub(notations, "tied", type=kind)
    if pos is not None:
        tech = _sub(notations, "technical")
        _sub(tech, "string", str(pos[1]))
        _sub(tech, "fret", str(pos[2]))


def to_musicxml(tab: Tab) -> str:
    bar = tab.beats_per_bar * TICKS_PER_BEAT
    notes = sorted(tab.notes, key=lambda n: n.tick)
    events = []                                     # (start, end, midi, string, fret), monophonic
    for i, n in enumerate(notes):
        end = n.tick + n.length
        if i + 1 < len(notes):
            end = min(end, notes[i + 1].tick)
        if end > n.tick:
            events.append((n.tick, end, n.midi, n.string, n.fret))
    total = max((e[1] for e in events), default=0)
    n_bars = max(1, -(-total // bar))

    chords: list[tuple[int, str]] = []
    for tick, name in sorted(((c.tick, chord_root(c.name)) for c in tab.chords)):
        if not chords or chords[-1][1] != name:
            chords.append((tick, name))

    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", tab.title)
    part_list = _sub(root, "part-list")
    score_part = _sub(part_list, "score-part", id="P1")
    _sub(score_part, "part-name", "Bass")
    part = _sub(root, "part", id="P1")

    measures = [_sub(part, "measure", number=i + 1) for i in range(n_bars)]
    _attributes(measures[0], tab)
    _tempo(measures[0], tab.bpm)

    def write(start: int, end: int, pos: tuple[int, int, int] | None) -> None:
        """Fill [start, end) with notes/rests, splitting and tying across bar lines."""
        first = True
        while start < end:
            b = start // bar
            stop = min(end, (b + 1) * bar)
            pieces = _split(stop - start)
            for j, (size, dur, dotted) in enumerate(pieces):
                last = stop == end and j == len(pieces) - 1
                tie = None
                if pos is not None:
                    tie = ("start" if first else "both") if not last else (None if first else "stop")
                while chords and start <= chords[0][0] < start + size:
                    _harmony(measures[b], chords.pop(0)[1])
                _note(measures[b], size, _TYPES[dur], dotted, pos, tie)
                first = False
                start += size

    cursor = 0
    for start, end, midi, string, fret in events:
        while chords and chords[0][0] < cursor:
            chords.pop(0)
        if start > cursor:
            write(cursor, start, None)
        write(start, end, (midi, string, fret))
        cursor = end
    if cursor < n_bars * bar:
        write(cursor, n_bars * bar, None)

    ET.indent(root)
    head = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
            '"http://www.musicxml.org/dtds/partwise.dtd">\n')
    return head + ET.tostring(root, encoding="unicode") + "\n"
