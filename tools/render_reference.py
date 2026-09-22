"""Render a reference JSON as an AlphaTab page so a human can check the transcription.

usage: uv run python tools/render_reference.py tools/reference/X.json   -> jobs/_refs/X.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bass_tab.__main__ import write_viewer  # noqa: E402
from bass_tab.contracts import Tab, TabNote, TUNINGS  # noqa: E402
from bass_tab.stage6_render import to_alphatex  # noqa: E402


def main(path: str) -> Path:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    tuning = spec.get("tuning") or list(TUNINGS["standard"])
    notes = [TabNote(tuning[s - 1] + f, t, n, s, f) for t, s, f, n in spec["notes"]]
    tab = Tab(f"REFERENCE {Path(path).stem}", spec.get("bpm", 120), 4, notes, tuning)
    out = ROOT / "jobs" / "_refs" / (Path(path).stem + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_viewer(to_alphatex(tab), out)
    print(out)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
