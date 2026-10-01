"""Score detected chord roots against the roots printed on a reference sheet.

usage: uv run python tools/eval_chords.py jobs/<id> tools/reference/<id>_chords.json
reference json: {"start_s": .., "bar_s": .., "roots": ["A", "G#", ...]}  (one root per bar)

A bar counts as correct when the root that covers most of it matches. Covering the bar instead
of sampling one instant keeps the score independent of where our segments fall inside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bass_tab.contracts import TAB_JSON, load_tab  # noqa: E402
from bass_tab.stage6_render import chord_root  # noqa: E402


def main(job_dir: str, ref_path: str) -> tuple[int, int]:
    spec = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    chords = sorted(load_tab(Path(job_dir) / TAB_JSON).chords, key=lambda c: c.start)
    spans = [(c.start, n.start if (n := chords[i + 1] if i + 1 < len(chords) else None) else 1e9,
              chord_root(c.name)) for i, c in enumerate(chords)]
    hits = []
    for i, want in enumerate(spec["roots"]):
        t0 = spec["start_s"] + i * spec["bar_s"]
        t1 = t0 + spec["bar_s"]
        cover: dict[str, float] = {}
        for start, end, root in spans:
            overlap = min(end, t1) - max(start, t0)
            if overlap > 0:
                cover[root] = cover.get(root, 0.0) + overlap
        got = max(cover, key=cover.get) if cover else None
        hits.append((want, got, want == got))
    ok = sum(h[2] for h in hits)
    print(f"마디별 근음: {ok}/{len(hits)} 정답")
    for i, (want, got, good) in enumerate(hits):
        print(f"  마디 {i + 1}: 악보 {want:3s} 검출 {str(got):4s} {'OK' if good else '<--'}")
    return ok, len(hits)


if __name__ == "__main__":
    main(*sys.argv[1:3])
