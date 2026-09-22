"""Score a job's tab.json against a hand-transcribed reference excerpt.

usage: uv run python tools/eval_reference.py jobs/awBbD1fxwio tools/reference/awBbD1fxwio_intro.json

The bar offset between the job and the reference is found automatically (the shift with
the most onset+pitch matches). A detected note "matches" a reference note when the midi is
equal and the tick is equal (exact) or within 1 tick (±1 = one 16th).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bass_tab.contracts import NOTES_JSON, OPEN_MIDI, TAB_JSON, load_notes, load_tab  # noqa: E402

START_WINDOW_S = 3.0


def score(det: list[tuple[int, int, int, int, int]], ref, ghosts, tol: int) -> dict:
    """det/ref rows: (tick, midi, string, fret, length)."""
    lo, hi = ref[0][0] - tol, ref[-1][0] + tol
    window = [d for d in det if lo <= d[0] <= hi and all(abs(d[0] - g) > 1 for g in ghosts)]
    used, hits = set(), []
    for r in ref:
        best = None
        for i, d in enumerate(window):
            if i not in used and d[1] == r[1] and abs(d[0] - r[0]) <= tol:
                if best is None or abs(d[0] - r[0]) < abs(window[best][0] - r[0]):
                    best = i
        if best is not None:
            used.add(best)
            hits.append((r, window[best]))
    n_ref, n_det, n_hit = len(ref), len(window), len(hits)
    p = n_hit / n_det if n_det else 0.0
    r = n_hit / n_ref
    return {
        "ref": n_ref, "detected": n_det, "matched": n_hit,
        "precision": round(p, 3), "recall": round(r, 3),
        "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
        "same_string_fret": sum(h[0][2:4] == h[1][2:4] for h in hits),
        "same_length": sum(h[0][4] == h[1][4] for h in hits),
    }


def pitch_report(det, ref, ghosts) -> dict:
    """For reference notes with any detected onset within ±1 tick: pitch correctness."""
    near = [(r, min((d for d in det if abs(d[0] - r[0]) <= 1), key=lambda d: abs(d[0] - r[0]),
                    default=None)) for r in ref]
    found = [(r, d) for r, d in near if d is not None]
    return {
        "onset_found": len(found),
        "pitch_correct": sum(r[1] == d[1] for r, d in found),
        "octave_error": sum(abs(r[1] - d[1]) == 12 for r, d in found),
        "semitone_off": sum(abs(r[1] - d[1]) == 1 for r, d in found),
    }


def evaluate(tab_notes, spec: dict, starts: list[float] | None = None) -> dict:
    """starts: detected note onset seconds (same order as tab_notes). With spec["start_s"]
    (approx. time of reference tick 0) the alignment search stays within +-START_WINDOW_S,
    so a repeated section elsewhere in the song can't be picked instead."""
    tuning = spec.get("tuning") or [OPEN_MIDI[s] for s in (1, 2, 3, 4)]
    ref = [(t, tuning[s - 1] + f, s, f, n) for t, s, f, n in spec["notes"]]
    raw = [(n.tick, n.midi, n.string, n.fret, n.length) for n in tab_notes]

    def shifted(off):
        return [(t - off, m, s, f, n) for t, m, s, f, n in raw]

    if starts is not None and "start_s" in spec:
        near = [n.tick for n, s in zip(tab_notes, starts) if abs(s - spec["start_s"]) <= START_WINDOW_S]
        offsets = range(min(near) - 8, max(near) + 9) if near else range(0)
    else:
        offsets = range(-64, max((n.tick for n in tab_notes), default=0) + 1)
    off = max(offsets, key=lambda o: score(shifted(o), ref, spec["ghost_ticks"], 0)["matched"])
    det = shifted(off)
    return {"offset_ticks": off,
            "exact": score(det, ref, spec["ghost_ticks"], 0),
            "within_1_tick": score(det, ref, spec["ghost_ticks"], 1),
            "pitch": pitch_report(det, ref, spec["ghost_ticks"])}


def main(job_dir: str, ref_path: str) -> dict:
    spec = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    starts = [n.start for n in load_notes(Path(job_dir) / NOTES_JSON)]
    out = evaluate(load_tab(Path(job_dir) / TAB_JSON).notes, spec, starts)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:3])
