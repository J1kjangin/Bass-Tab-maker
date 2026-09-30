from xml.etree import ElementTree as ET

from bass_tab.contracts import Chord, Tab, TabNote
from bass_tab.musicxml import to_musicxml


def parse(tab):
    return ET.fromstring(to_musicxml(tab).split("?>", 1)[1].split(">", 1)[1])


def test_tab_staff_header_and_note():
    tab = Tab("My Song", 120.0, 4, [TabNote(33, 0, 4, 3, 0)])
    root = parse(tab)
    at = root.find("part/measure/attributes")
    assert at.findtext("divisions") == "4" and at.findtext("clef/sign") == "TAB"
    assert at.findtext("staff-details/staff-lines") == "4"
    tunings = [(t.get("line"), t.findtext("tuning-step"), t.findtext("tuning-octave"))
               for t in at.findall("staff-details/staff-tuning")]
    assert tunings[0] == ("4", "G", "2") and tunings[-1] == ("1", "E", "1")  # string 1 on top line
    assert root.findtext("part/measure/direction/direction-type/metronome/per-minute") == "120"
    note = root.find("part/measure/note")
    assert (note.findtext("pitch/step"), note.findtext("pitch/octave")) == ("A", "1")
    assert note.findtext("duration") == "4" and note.findtext("type") == "quarter"
    assert note.findtext("notations/technical/string") == "3"
    assert note.findtext("notations/technical/fret") == "0"


def test_rest_tie_across_bar_and_chord_root():
    # one note from tick 12 over the bar line, after 12 ticks of rest, with a chord at tick 12
    tab = Tab("T", 100.0, 4, [TabNote(35, 12, 8, 3, 2)], chords=[Chord("G#m7", 0.0, 12)])
    root = parse(tab)
    measures = root.findall("part/measure")
    assert len(measures) == 2
    assert measures[0].findtext("harmony/root/root-step") == "G"
    assert measures[0].findtext("harmony/root/root-alter") == "1"
    first = [n for n in measures[0].findall("note") if n.find("rest") is None][-1]
    assert first.find("tie").get("type") == "start"
    second = measures[1].find("note")
    assert second.find("tie").get("type") == "stop"
    assert sum(int(n.findtext("duration")) for m in measures for n in m.findall("note")) == 32
