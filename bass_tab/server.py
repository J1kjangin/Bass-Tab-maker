"""Job server (design doc 0.4): uv run uvicorn bass_tab.server:app --host 127.0.0.1 --port 8000

POST /api/jobs            form: url=... | file=<audio>, tuning=standard|drop-d -> {job_id, status}
GET  /api/jobs/{id}       -> {status: queued|processing|done|failed, stage, progress, error}
GET  /api/jobs/{id}/result -> {alphatex, tab}

One worker thread runs jobs one at a time: a single job already saturates every CPU core, and
Celery (no Windows support since 4.x) / Redis (Windows only via Docker) don't fit this machine.
ponytail: in-process queue; jobs waiting at a server restart are marked failed. Swap the worker
for Celery + Redis on a Linux deploy; the HTTP API stays the same.
"""
from __future__ import annotations

import json
import queue
import re
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import __main__ as cli
from .contracts import TAB_JSON, TAB_TEX, TUNINGS, PipelineError

JOBS = cli.ROOT / "jobs"
APP_PAGE = cli.ROOT / "web" / "app.html"
STATUS = "status.json"
MAX_UPLOAD = 200 * 2**20
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm", ".mp4"}
ID_RE = re.compile(r"^[\w.-]{1,64}$")

_queue: queue.Queue = queue.Queue()
_lock = threading.Lock()


def _write(job_id: str, **fields) -> dict:
    with _lock:
        path = JOBS / job_id / STATUS
        cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        cur.update(fields)
        path.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
        return cur


def _read(job_id: str) -> dict:
    path = JOBS / job_id / STATUS
    if not ID_RE.match(job_id) or job_id.startswith(".") or not path.exists():
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return json.loads(path.read_text(encoding="utf-8"))


def _worker() -> None:
    device = cli.pick_device("auto")
    while (item := _queue.get()) is not None:  # None = server shutting down
        job_id, source, tuning = item
        _write(job_id, status="processing")
        try:
            cli.run(source, JOBS / job_id, "htdemucs", device, False, tuning,
                    on_stage=lambda stage, pct: _write(job_id, stage=stage, progress=pct))
            _write(job_id, status="done", stage="done", progress=100)
        except PipelineError as e:
            _write(job_id, status="failed", error=str(e))
        except Exception as e:  # keep the worker alive for the next job
            traceback.print_exc()
            _write(job_id, status="failed", error=f"내부 오류: {type(e).__name__}: {e}")


def _fail_interrupted() -> None:
    for path in JOBS.glob(f"*/{STATUS}"):
        if json.loads(path.read_text(encoding="utf-8")).get("status") in ("queued", "processing"):
            _write(path.parent.name, status="failed", error="서버가 재시작되어 중단되었습니다. 다시 요청하세요.")


@asynccontextmanager
async def lifespan(_app):
    JOBS.mkdir(exist_ok=True)
    _fail_interrupted()
    threading.Thread(target=_worker, daemon=True).start()
    yield
    _queue.put(None)


app = FastAPI(title="Bass Tab Maker", lifespan=lifespan)


@app.get("/")
def page():
    return FileResponse(APP_PAGE)


@app.post("/api/jobs")
async def create_job(url: str = Form(""), tuning: str = Form("standard"),
                     file: UploadFile | None = File(None)):
    url = url.strip()
    if bool(url) == bool(file and file.filename):
        raise HTTPException(400, "링크 또는 오디오 파일 중 하나만 보내 주세요")
    if tuning not in TUNINGS:
        raise HTTPException(400, f"지원하지 않는 튜닝입니다: {tuning}")

    if url:
        if not url.startswith(("http://", "https://")):
            raise HTTPException(400, "http(s) 링크만 지원합니다")
        job_id = cli.job_id_for(url)
        if (JOBS / job_id / STATUS).exists() and _read(job_id)["status"] in ("queued", "processing"):
            return {"job_id": job_id, **_read(job_id)}  # same link already in progress
        (JOBS / job_id).mkdir(parents=True, exist_ok=True)
        source = url
    else:
        ext = Path(file.filename).suffix.lower()
        if ext not in AUDIO_EXT:
            raise HTTPException(400, f"지원하는 오디오 형식: {', '.join(sorted(AUDIO_EXT))}")
        job_id = uuid.uuid4().hex[:12]
        (JOBS / job_id / "upload").mkdir(parents=True)
        # the file stem becomes the tab title (Stage 0), so keep the user's name, sanitized
        dest = JOBS / job_id / "upload" / re.sub(r"[^\w.-]", "_", Path(file.filename).name).lstrip(".")
        size = 0
        with dest.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    out.close()
                    dest.unlink()
                    raise HTTPException(413, f"파일은 {MAX_UPLOAD >> 20}MB 이하만 받습니다")
                out.write(chunk)
        source = str(dest)

    status = _write(job_id, status="queued", stage="queued", progress=0, error=None, tuning=tuning)
    _queue.put((job_id, source, tuning))
    return {"job_id": job_id, **status}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _read(job_id)


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    if _read(job_id)["status"] != "done":
        raise HTTPException(409, "아직 완료되지 않았습니다")
    d = JOBS / job_id
    return {"alphatex": (d / TAB_TEX).read_text(encoding="utf-8"),
            "tab": json.loads((d / TAB_JSON).read_text(encoding="utf-8"))}
