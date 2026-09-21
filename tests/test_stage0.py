import numpy as np
import soundfile as sf

from bass_tab.contracts import META, MIX_WAV, SR, load_meta
from bass_tab.stage0_input import fetch


def test_local_file_to_mix_wav(tmp_path):
    src = tmp_path / "my song.wav"
    t = np.arange(22050 * 2) / 22050
    sf.write(src, 0.5 * np.sin(2 * np.pi * 110 * t), 22050)  # mono 22.05 kHz
    job = tmp_path / "job"

    meta = fetch(str(src), job)

    info = sf.info(job / MIX_WAV)
    assert (info.samplerate, info.channels, info.subtype) == (SR, 2, "PCM_16")
    x, _ = sf.read(job / MIX_WAV, dtype="int16")
    assert np.array_equal(x[:, 0], x[:, 1])
    assert meta.title == "my song"
    assert abs(meta.duration - 2.0) < 0.01
    assert load_meta(job / META) == meta
