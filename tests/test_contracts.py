import numpy as np

from bass_tab.contracts import (Beats, Meta, Note, Pitch, Tab, TabNote, load_beats, load_meta,
                                load_notes, load_tab, save_json)


def test_roundtrip(tmp_path):
    meta = Meta("x.wav", "t", 1.5, ["w"])
    save_json(meta, tmp_path / "m.json")
    assert load_meta(tmp_path / "m.json") == meta

    beats = Beats([0.5, 1.0], [0.5], 120.0)
    save_json(beats, tmp_path / "b.json")
    assert load_beats(tmp_path / "b.json") == beats

    notes = [Note(40, 0.5, 0.7, 0, 2)]
    save_json(notes, tmp_path / "n.json")
    assert load_notes(tmp_path / "n.json") == notes

    tab = Tab("t", 120.0, 4, [TabNote(40, 0, 2, 3, 7)])
    save_json(tab, tmp_path / "t.json")
    assert load_tab(tmp_path / "t.json") == tab

    p = Pitch(np.array([0.0, 0.01]), np.array([55.0, np.nan]), np.array([0.9, 0.1]))
    p.save(tmp_path / "p.npz")
    q = Pitch.load(tmp_path / "p.npz")
    assert np.allclose(q.f0, p.f0, equal_nan=True) and np.array_equal(q.time, p.time)
