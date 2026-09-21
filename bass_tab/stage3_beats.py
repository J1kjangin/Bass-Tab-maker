"""Stage 3: beat/downbeat tracking on the original mix with beat_this (CPJKU, MIT)."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from .contracts import BEATS_JSON, MIX_WAV, Beats, PipelineError, save_json


@lru_cache(maxsize=2)
def _model(device: str):
    from beat_this.inference import File2Beats
    return File2Beats(checkpoint_path="final0", device=device, dbn=False)


def track(job_dir: Path, device: str = "cpu") -> Beats:
    job_dir = Path(job_dir)
    beats, downbeats = _model(device)(str(job_dir / MIX_WAV))
    beats = np.sort(np.asarray(beats, dtype=float))
    if len(beats) < 2:
        raise PipelineError("비트를 검출하지 못했습니다")

    # snap each downbeat to its nearest beat so downbeats is a subset of beats
    idx = sorted({int(np.argmin(np.abs(beats - d))) for d in np.asarray(downbeats, dtype=float)})
    gaps = np.diff(idx)
    beats_per_bar = Counter(gaps.tolist()).most_common(1)[0][0] if len(gaps) else 4

    # beat_this outputs times on a 20 ms frame grid, so the plain median of diffs is biased
    # (128 BPM -> 130.4). Use the median as an outlier gate, then average the inliers.
    d = np.diff(beats)
    med = np.median(d)
    period = d[np.abs(d - med) <= 0.15 * med].mean()

    result = Beats(
        beats=[float(b) for b in beats],
        downbeats=[float(beats[i]) for i in idx],
        bpm=float(60.0 / period),
        beats_per_bar=int(beats_per_bar),
    )
    save_json(result, job_dir / BEATS_JSON)
    return result
