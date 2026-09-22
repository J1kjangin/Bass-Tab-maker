"""Stage 2: bass.wav -> frame-level f0 (torchcrepe) -> pitch.npz."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchcrepe
from torchcrepe.core import PITCH_BINS, infer, postprocess, preprocess

from .contracts import BASS_WAV, PITCH_HOP_S, PITCH_NPZ, Pitch

CREPE_SR = 16000
HOP = int(round(PITCH_HOP_S * CREPE_SR))  # 160 samples
FMIN = 32.7        # below ~31.7 Hz torchcrepe returns -inf for every frame (CLAUDE.md)
FMAX = 400.0
# "full" ~2.1 s per audio-second on this CPU, "tiny" ~0.18 s. tiny matched full on the first
# test song, but on a second song (weak fundamental, strong 2nd harmonic) it marked only 17%
# of frames voiced vs 50% for full and missed loud G1 notes entirely -> "full".
# torchcrepe ships only these two capacities.
MODEL = "full"
CONF_THRESHOLD = 0.5   # design doc: periodicity < 0.5 -> unvoiced
SILENCE_DB = -50.0     # frame RMS this far below the loudest frame -> unvoiced
# torchcrepe runs the network and viterbi per batch, so this is also the memory chunk.
# ~512 frames (5 s) keeps peak RAM around 1-2 GB on CPU for the "full" model.
BATCH = 512

# No octave clamp (design doc's 41.2-400 Hz fold): fmin/fmax already restrict the
# decoder to 32.7-400 Hz, and folding a sub-41 Hz reading up an octave turns a
# drop-D / 5-string low note into a wrong note instead of a correctable one.


def _decode(audio: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    """One network pass, two decodings (torchcrepe.predict only offers one).

    Viterbi alone got stuck at the fmin bin after silences and missed short loud notes
    (confidence ~0 on a clear E2); cutting the same audio at another point found them at 0.95.
    So: confidence = best bin's probability (decoder-independent), pitch = viterbi, except
    where it is > half a semitone from the per-frame weighted argmax (a stuck path).
    4 reference songs, exact onset+pitch F1: viterbi-only 0.774 -> this 0.803; no song got worse.
    """
    vit, arg, conf = [], [], []
    with torch.no_grad():
        for frames in preprocess(audio, CREPE_SR, HOP, BATCH, device, True):
            probs = infer(frames, MODEL, device).reshape(1, -1, PITCH_BINS).transpose(1, 2)
            vit.append(postprocess(probs.clone(), FMIN, FMAX, torchcrepe.decode.viterbi))
            f, p = postprocess(probs, FMIN, FMAX, torchcrepe.decode.weighted_argmax,
                               return_periodicity=True)
            arg.append(f); conf.append(p)
    vf, af = (torch.cat(x, 1)[0].cpu().numpy().astype(np.float64) for x in (vit, arg))
    stuck = np.abs(librosa.hz_to_midi(vf) - librosa.hz_to_midi(af)) > 0.5
    return np.where(stuck, af, vf), torch.cat(conf, 1)[0].cpu().numpy().astype(np.float64)


def track(job_dir: Path, device: str = "cpu") -> Pitch:
    job_dir = Path(job_dir)
    audio, sr = sf.read(job_dir / BASS_WAV, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != CREPE_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CREPE_SR)

    f0, conf = _decode(torch.from_numpy(np.ascontiguousarray(audio))[None], device)

    # Same framing as torchcrepe (pad=True): 1024 window centered on n * HOP.
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=HOP, center=True)[0]
    rms = rms[: len(f0)]
    db = 20 * np.log10(np.maximum(rms, 1e-10) / max(rms.max(), 1e-10))

    f0[(conf < CONF_THRESHOLD) | (db < SILENCE_DB)] = np.nan
    pitch = Pitch(time=np.arange(len(f0)) * PITCH_HOP_S, f0=f0, confidence=conf)
    pitch.save(job_dir / PITCH_NPZ)
    return pitch
