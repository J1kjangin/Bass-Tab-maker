"""Score a job's note sequence (pitch order only) against a hand-read reference.

For songs where the on-screen sheet gives reliable fret numbers but the rhythm is hard to read,
this checks what the pipeline heard, not when: the reference pitch sequence is aligned to the
detected notes with an edit distance over every possible start, and the best window is reported.

usage: uv run python tools/eval_sequence.py jobs/<id> tools/reference/<name>.json
reference json: {"pitches": [...], "chord_roots": ["C#", ...]}  (chord_roots optional)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bass_tab.contracts import load_notes, load_tab  # noqa: E402
from bass_tab.stage6_render import chord_root  # noqa: E402


def align(ref: list[int], det: list[int]) -> tuple[int, int, int, int, int]:
    """Levenshtein between the two sequences -> (distance, matches, subs, missing, extra)."""
    n, m = len(ref), len(det)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], back[i][0] = i, "d"
    for j in range(1, m + 1):
        dp[0][j], back[0][j] = j, "i"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = ref[i - 1] == det[j - 1]
            best = (dp[i - 1][j - 1] + (0 if same else 1), "m" if same else "s")
            for cost, op in ((dp[i - 1][j] + 1, "d"), (dp[i][j - 1] + 1, "i")):
                if cost < best[0]:
                    best = (cost, op)
            dp[i][j], back[i][j] = best
    i, j, counts = n, m, {"m": 0, "s": 0, "d": 0, "i": 0}
    while i or j:
        op = back[i][j]
        counts[op] += 1
        i -= op in "msd"
        j -= op in "msi"
    return dp[n][m], counts["m"], counts["s"], counts["d"], counts["i"]


def best_window(ref: list[int], det: list[int]) -> tuple[int, int, tuple[int, int, int, int]]:
    """Slide over the detected notes; return (start, end, score) of the closest window."""
    best = (10**9, 0, 0, None)
    for start in range(max(1, len(det) - len(ref) + 1)):
        # the window may hold up to as many spurious notes again as the reference has
        for extra in range(0, len(ref) + 1, 2):
            end = min(len(det), start + len(ref) + extra)
            score = align(ref, det[start:end])
            if score[0] < best[0]:
                best = (score[0], start, end, score)
    return best[1], best[2], best[3]


def main(job_dir: str, ref_path: str) -> None:
    spec = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    notes = load_notes(Path(job_dir) / "notes.json")
    det = [n.midi for n in notes]
    start, end, (dist, matches, subs, missing, extra) = best_window(spec["pitches"], det)
    n = len(spec["pitches"])
    print(f"음높이 순서: 정답 {n}개 중 일치 {matches} ({matches / n:.0%}), "
          f"다른 음 {subs}, 놓침 {missing}, 없는 음 추가 {extra}")
    print(f"  맞춰진 구간: 노트 {start}~{end - 1}, {notes[start].start:.1f}s ~ {notes[end - 1].end:.1f}s")
    roots = spec.get("chord_roots")
    if roots:
        tab = load_tab(Path(job_dir) / "tab.json")
        lo, hi = notes[start].start, notes[end - 1].end
        got = []
        for c in tab.chords:
            if lo - 1.0 <= c.start <= hi:
                r = chord_root(c.name)
                if not got or got[-1] != r:
                    got.append(r)
        hits = sum(1 for r in roots if r in got)
        print(f"코드 근음: 악보 {roots}")
        print(f"          검출 {got}  → 악보에 있는 근음 중 {hits}/{len(roots)} 등장")


if __name__ == "__main__":
    main(*sys.argv[1:3])
