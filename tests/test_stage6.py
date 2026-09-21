from fractions import Fraction

from bass_tab.contracts import Tab, TabNote
from bass_tab.stage6_render import to_alphatex

HEADER = (
    '\\title "T"\n'
    '\\track "Bass" { instrument 33 }\n'
    "\\staff { score tabs }\n"
    "\\tuning (G2 D2 A1 E1)\n"
    "\\clef F4\n"
    "\\ts (4 4)\n"
    "\\tempo 100\n"
)


def test_exact_output_with_rest_and_tie_across_bar():
    tab = Tab("T", 100.0, 4, [TabNote(28, 0, 4, 4, 0), TabNote(33, 6, 14, 3, 0)])
    assert to_alphatex(tab) == HEADER + "0.4.4 r.8 0.3.2 -.3.8 |\n-.3.4 r.2{d}\n"


def test_empty_tab_is_one_rest_bar():
    assert to_alphatex(Tab("T", 100.0, 4, [])) == HEADER + "r.1\n"


def bar_beats(tex):
    body = tex.strip().split("\n\\tempo")[1].split("\n", 1)[1]
    out = []
    for bar in body.split("|"):
        total = Fraction(0)
        for beat in bar.split():
            dur = beat.split(".")[-1]
            base = Fraction(4, int(dur.removesuffix("{d}")))  # in quarter-note beats
            total += base * Fraction(3, 2) if dur.endswith("{d}") else base
        out.append(total)
    return out


def test_bars_are_full_in_3_4():
    notes = [TabNote(33, 1, 5, 3, 0), TabNote(35, 7, 13, 3, 2), TabNote(38, 30, 3, 2, 0)]
    beats = bar_beats(to_alphatex(Tab("T", 90.0, 3, notes)))
    assert beats == [3, 3, 3]  # last note ends at tick 33 -> 3 bars of 12 ticks


def test_overlapping_notes_are_clipped():
    notes = [TabNote(33, 0, 8, 3, 0), TabNote(35, 4, 4, 3, 2)]
    assert to_alphatex(Tab("T", 100.0, 4, notes)).endswith("0.3.4 2.3.4 r.2\n")
