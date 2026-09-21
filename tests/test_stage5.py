from bass_tab.contracts import OPEN_MIDI, Note
from bass_tab.stage5_frets import assign


def notes(*midis):
    return [Note(midi=m, start=i * 0.5, end=i * 0.5 + 0.5, tick=i * 4, length=4) for i, m in enumerate(midis)]


def positions(tab):
    for t in tab:
        assert OPEN_MIDI[t.string] + t.fret == t.midi
    return [(t.string, t.fret) for t in tab]


def test_empty():
    assert assign([]) == []


def test_open_strings():
    assert positions(assign(notes(28, 33, 38, 43))) == [(4, 0), (3, 0), (2, 0), (1, 0)]


def test_chromatic_run_stays_in_position():
    pos = positions(assign(notes(40, 41, 42, 43, 44)))
    assert pos == [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]


def test_out_of_range_is_octave_shifted():
    tab = assign(notes(24, 70))  # C1 below low E, A#4 above G string fret 20
    assert [t.midi for t in tab] == [36, 58]
    positions(tab)
    assert [(t.tick, t.length) for t in tab] == [(0, 4), (4, 4)]
