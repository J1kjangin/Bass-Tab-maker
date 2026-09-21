import numpy as np
import pytest
import soundfile as sf

from bass_tab.contracts import MIX_WAV, SR, PipelineError
from bass_tab.stage1_separate import MIN_RMS_DB, rms_db, separate


def _write_mix(job, x):
    job.mkdir()
    sf.write(job / MIX_WAV, np.stack([x, x], axis=1), SR, subtype="PCM_16")


def test_bass_extracted(tmp_path):
    t = np.arange(SR * 5) / SR
    saw = 0.4 * (2 * ((55 * t) % 1) - 1)
    clicks = np.zeros_like(t)
    clicks[:: SR // 2] = 0.8
    job = tmp_path / "job"
    _write_mix(job, saw + clicks)

    out = separate(job)

    assert out == job / "bass.wav" and out.exists()
    assert rms_db(out) > MIN_RMS_DB
    assert set(job.iterdir()) == {out, job / MIX_WAV}  # temp output removed


def test_silent_mix_rejected(tmp_path):
    rng = np.random.default_rng(0)
    job = tmp_path / "job"
    _write_mix(job, 1e-5 * rng.standard_normal(SR * 4))
    with pytest.raises(PipelineError, match="베이스"):
        separate(job)
