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


MIN_SHORT_SHARE = 0.2   # the short-interval cluster must be this common to count as the beat
TOL = 0.15              # relative tolerance for "same interval" / "integer multiple"


def fill_missing(beats: np.ndarray) -> np.ndarray:
    """beat_this (dbn=False) can drop beats: on a ~190 BPM song 62% of intervals were 2 beats
    long, which made the whole grid half-time and irregular. If the shortest interval cluster
    is common enough, split every interval that is ~k times it into k equal beats."""
    d = np.diff(beats)
    p0 = np.percentile(d, 10)
    short = np.abs(d - p0) <= TOL * p0
    if short.mean() < MIN_SHORT_SHARE:
        return beats
    period = np.median(d[short])
    out = [beats[0]]
    for a, b in zip(beats[:-1], beats[1:]):
        k = round((b - a) / period)
        if k >= 2 and abs((b - a) / period - k) <= TOL:
            out.extend(a + (b - a) * np.arange(1, k) / k)
        out.append(b)
    return np.asarray(out)


def _beats_per_bar(gaps: np.ndarray) -> int:
    """Beats between consecutive downbeats. Missed downbeats make some gaps 2x a bar
    (Drop-D test song: 4s and 8s, 8 most common), so take the smallest common count that
    at least 80% of gaps are multiples of."""
    gaps = gaps[gaps >= 2]
    if not len(gaps):
        return 4
    counts = Counter(gaps.tolist())
    for c in sorted(k for k, n in counts.items() if n / len(gaps) >= MIN_SHORT_SHARE):
        if np.mean(gaps % c == 0) >= 0.8:
            return int(c)
    return int(counts.most_common(1)[0][0])


def track(job_dir: Path, device: str = "cpu") -> Beats:
    job_dir = Path(job_dir)
    beats, downbeats = _model(device)(str(job_dir / MIX_WAV))
    beats = np.sort(np.asarray(beats, dtype=float))
    if len(beats) < 2:
        raise PipelineError("비트를 검출하지 못했습니다")
    beats = fill_missing(beats)

    # snap each downbeat to its nearest beat so downbeats is a subset of beats
    idx = sorted({int(np.argmin(np.abs(beats - d))) for d in np.asarray(downbeats, dtype=float)})
    beats_per_bar = _beats_per_bar(np.diff(idx))

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
