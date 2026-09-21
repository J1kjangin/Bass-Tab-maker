"""Stage 2: bass.wav -> frame-level f0 (torchcrepe) -> pitch.npz."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchcrepe

from .contracts import BASS_WAV, PITCH_HOP_S, PITCH_NPZ, Pitch

CREPE_SR = 16000
HOP = int(round(PITCH_HOP_S * CREPE_SR))  # 160 samples
FMIN = 32.7        # below ~31.7 Hz torchcrepe returns -inf for every frame (CLAUDE.md)
FMAX = 400.0
# "full" ~2.5-3.5 s per audio-second on this CPU; "tiny" ~0.15 s and matched full on the
# synthetic test. Kept "full" (accuracy first) until real-audio comparison says otherwise.
MODEL = "full"
CONF_THRESHOLD = 0.5   # design doc: periodicity < 0.5 -> unvoiced
SILENCE_DB = -50.0     # frame RMS this far below the loudest frame -> unvoiced
# torchcrepe runs the network and viterbi per batch, so this is also the memory chunk.
# ~512 frames (5 s) keeps peak RAM around 1-2 GB on CPU for the "full" model.
BATCH = 512

# No octave clamp (design doc's 41.2-400 Hz fold): fmin/fmax already restrict the
# decoder to 32.7-400 Hz, and folding a sub-41 Hz reading up an octave turns a
# drop-D / 5-string low note into a wrong note instead of a correctable one.


def track(job_dir: Path, device: str = "cpu") -> Pitch:
    job_dir = Path(job_dir)
    audio, sr = sf.read(job_dir / BASS_WAV, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != CREPE_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CREPE_SR)

    f0, conf = torchcrepe.predict(
        torch.from_numpy(np.ascontiguousarray(audio))[None],
        CREPE_SR,
        hop_length=HOP,
        fmin=FMIN,
        fmax=FMAX,
        model=MODEL,
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        batch_size=BATCH,
        device=device,
        pad=True,
    )
    f0 = f0[0].cpu().numpy().astype(np.float64)
    conf = conf[0].cpu().numpy().astype(np.float64)

    # Same framing as torchcrepe (pad=True): 1024 window centered on n * HOP.
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=HOP, center=True)[0]
    rms = rms[: len(f0)]
    db = 20 * np.log10(np.maximum(rms, 1e-10) / max(rms.max(), 1e-10))

    f0[(conf < CONF_THRESHOLD) | (db < SILENCE_DB)] = np.nan
    pitch = Pitch(time=np.arange(len(f0)) * PITCH_HOP_S, f0=f0, confidence=conf)
    pitch.save(job_dir / PITCH_NPZ)
    return pitch
