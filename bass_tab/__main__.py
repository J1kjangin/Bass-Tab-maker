"""Run the whole pipeline synchronously: python -m bass_tab <url|file>.

Stages whose output file already exists in the job dir are skipped (use --force to redo),
so a failed or interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import (stage0_input, stage1_separate, stage2_pitch, stage3_beats, stage4_notes,
               stage5_frets, stage6_render)
from .contracts import (BASS_WAV, BEATS_JSON, META, NOTES_JSON, PITCH_NPZ, TAB_JSON, TAB_TEX,
                        TUNINGS, PipelineError, Tab, load_beats, load_meta, load_notes, save_json)

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "web" / "index.html"


def job_id_for(source: str) -> str:
    if source.startswith(("http://", "https://")):
        u = urlparse(source)
        raw = parse_qs(u.query).get("v", [u.path.rstrip("/").rsplit("/", 1)[-1]])[0]
    else:
        raw = Path(source).stem
    return re.sub(r"[^\w.-]", "_", raw) or "job"


def pick_device(requested: str) -> str:
    """"auto" -> cuda only if a real op succeeds there. torch can report cuda as available
    and still fail every op (seen: cu126 torch on a driver that only supports CUDA 12.2)."""
    import torch
    if requested == "auto":
        try:
            torch.ones(1, device="cuda").add_(1).item()
            requested = "cuda"
        except Exception:
            requested = "cpu"
    if requested == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)  # default was 4 of 8; measured 14% faster
    return requested


def write_viewer(tex: str, out: Path) -> None:
    page = VIEWER.read_text(encoding="utf-8")
    page = re.sub(r'(<textarea id="tex">).*?(</textarea>)',
                  lambda m: m.group(1) + html.escape(tex) + m.group(2), page, flags=re.S)
    out.write_text(page, encoding="utf-8")


def run(source: str, job_dir: Path, sep_model: str, device: str, force: bool,
        tuning: str = "standard", on_stage=None) -> Path:
    """on_stage(stage, progress_percent) is called before each stage (used by the job server).
    Percentages follow measured CPU time shares: separation ~30%, pitch ~60%."""
    steps = [
        ("input", 0, META, lambda: stage0_input.fetch(source, job_dir)),
        ("separation", 2, BASS_WAV, lambda: stage1_separate.separate(job_dir, sep_model, device)),
        ("pitch", 32, PITCH_NPZ, lambda: stage2_pitch.track(job_dir, device)),
        ("beats", 92, BEATS_JSON, lambda: stage3_beats.track(job_dir, device)),
        ("notes", 97, NOTES_JSON, lambda: stage4_notes.segment(job_dir)),
    ]
    for name, pct, out, fn in steps:
        if on_stage:
            on_stage(name, pct)
        if (job_dir / out).exists() and not force:
            print(f"[stage {name}] skip ({out} exists)")
            continue
        t0 = time.perf_counter()
        fn()
        print(f"[stage {name}] {time.perf_counter() - t0:.1f}s")
    if on_stage:
        on_stage("render", 98)

    meta, beats = load_meta(job_dir / META), load_beats(job_dir / BEATS_JSON)
    open_midi = TUNINGS[tuning]
    tab = Tab(meta.title, beats.bpm, beats.beats_per_bar,
              stage5_frets.assign(load_notes(job_dir / NOTES_JSON), open_midi), list(open_midi))
    save_json(tab, job_dir / TAB_JSON)
    tex = stage6_render.to_alphatex(tab)
    (job_dir / TAB_TEX).write_text(tex, encoding="utf-8")
    write_viewer(tex, job_dir / "tab.html")
    print(f"[stage 5-6] {len(tab.notes)} notes, {beats.bpm:.1f} BPM, {beats.beats_per_bar}/4")
    for w in meta.warnings:
        print("warning:", w)
    return job_dir / "tab.html"


def main() -> None:
    p = argparse.ArgumentParser(prog="bass_tab", description="Audio link/file -> bass tab")
    p.add_argument("source", help="YouTube/music URL or local audio file")
    p.add_argument("--job-id", help="job folder name (default: video id / file name)")
    p.add_argument("--sep-model", default="htdemucs", help="Demucs model (htdemucs, htdemucs_6s)")
    p.add_argument("--device", default="auto", help="auto, cpu or cuda")
    p.add_argument("--force", action="store_true", help="redo stages even if outputs exist")
    p.add_argument("--tuning", default="standard", choices=sorted(TUNINGS))
    a = p.parse_args()
    job_dir = ROOT / "jobs" / (a.job_id or job_id_for(a.source))
    job_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    print("device:", device)
    try:
        out = run(a.source, job_dir, a.sep_model, device, a.force, a.tuning)
    except PipelineError as e:
        sys.exit(f"오류: {e}")
    print("tab:", out)


if __name__ == "__main__":
    main()
