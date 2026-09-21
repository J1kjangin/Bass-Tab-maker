"""Stage 0: URL or local file -> job_dir/mix.wav (44.1 kHz stereo 16-bit) + meta.json."""
from __future__ import annotations

import subprocess
from pathlib import Path

import soundfile as sf
import yt_dlp

from .contracts import META, MIX_WAV, SR, Meta, PipelineError, save_json

MAX_DURATION_S = 600


def _to_wav(src: Path, dst: Path) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise PipelineError(f"오디오 변환 실패: {r.stderr.strip()[-300:]}")


def _download(url: str, job_dir: Path) -> tuple[Path, str, float | None, list[str]]:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if info.get("is_live"):
            raise PipelineError("라이브 방송은 지원하지 않습니다")
        warnings = []
        dur = info.get("duration")
        if dur and dur > MAX_DURATION_S:
            warnings.append(f"음원 길이가 {dur:.0f}초로 10분을 넘습니다. 처리 시간이 오래 걸릴 수 있습니다.")
        opts |= {"format": "bestaudio/best", "outtmpl": str(job_dir / "download.%(ext)s")}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(info["requested_downloads"][0]["filepath"])
    except yt_dlp.utils.DownloadError as e:
        raise PipelineError(f"다운로드 실패: {str(e).removeprefix('ERROR: ')}") from e
    return path, info.get("title") or "untitled", dur, warnings


def fetch(source: str, job_dir: Path) -> Meta:
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    mix = job_dir / MIX_WAV
    if source.startswith(("http://", "https://")):
        path, title, _, warnings = _download(source, job_dir)
        try:
            _to_wav(path, mix)
        finally:
            for p in job_dir.glob("download.*"):
                p.unlink()
    else:
        path = Path(source)
        if not path.is_file():
            raise PipelineError(f"파일을 찾을 수 없습니다: {source}")
        _to_wav(path, mix)
        title, warnings = path.stem, []
    meta = Meta(source=source, title=title, duration=sf.info(mix).duration, warnings=warnings)
    save_json(meta, job_dir / META)
    return meta
