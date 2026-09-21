"""Stage 1: job_dir/mix.wav -> job_dir/bass.wav via Demucs (--two-stems=bass)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from demucs.separate import main as demucs_main

from .contracts import BASS_WAV, MIX_WAV, PipelineError

MIN_RMS_DB = -50.0


def rms_db(path: Path) -> float:
    x, _ = sf.read(path, dtype="float32")
    rms = float(np.sqrt(np.mean(np.square(x))))
    return 20 * np.log10(max(rms, 1e-12))


def separate(job_dir: Path, model: str = "htdemucs", device: str = "cpu") -> Path:
    """Write job_dir/bass.wav. Raises PipelineError if the bass is below MIN_RMS_DB.

    On gate failure bass.wav is kept on disk (useful for listening/debugging);
    callers must treat the PipelineError as "no usable bass".
    """
    job_dir = Path(job_dir)
    mix = job_dir / MIX_WAV
    out = job_dir / BASS_WAV
    with tempfile.TemporaryDirectory(dir=job_dir) as tmp:
        try:
            demucs_main(["--two-stems=bass", "-n", model, "-d", device, "-o", tmp, str(mix)])
        except SystemExit as e:  # demucs reports errors via sys.exit
            raise PipelineError(f"Demucs 분리 실패 (code {e.code})") from e
        # Demucs layout: {out}/{model}/{track stem}/bass.wav
        found = list(Path(tmp).glob("*/*/bass.wav"))
        if len(found) != 1:
            raise PipelineError(f"Demucs 출력에서 bass.wav를 찾지 못했습니다: {found}")
        shutil.move(found[0], out)
    if rms_db(out) < MIN_RMS_DB:
        raise PipelineError("베이스 트랙이 검출되지 않았습니다")
    return out
