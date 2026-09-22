import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bass_tab import server
from bass_tab.contracts import PipelineError, Tab, save_json


SOURCES = []


def fake_run(source, job_dir, sep_model, device, force, tuning, on_stage=None):
    SOURCES.append(source)
    if "fail" in source:
        raise PipelineError("라이브 방송은 지원하지 않습니다")
    for stage, pct in [("input", 0), ("pitch", 32), ("render", 98)]:
        on_stage(stage, pct)
    save_json(Tab("T", 120.0, 4, []), job_dir / "tab.json")
    (job_dir / "tab.alphatex").write_text("\\title \"T\"\nr.1\n", encoding="utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "JOBS", tmp_path)
    monkeypatch.setattr(server.cli, "run", fake_run)
    monkeypatch.setattr(server.cli, "pick_device", lambda _: "cpu")
    with TestClient(server.app) as c:
        yield c


def wait(client, job_id):
    for _ in range(100):
        s = client.get(f"/api/jobs/{job_id}").json()
        if s["status"] in ("done", "failed"):
            return s
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_url_job_runs_to_result(client):
    r = client.post("/api/jobs", data={"url": "https://youtu.be/abc123", "tuning": "drop-d"})
    assert r.status_code == 200 and r.json()["job_id"] == "abc123"
    assert wait(client, "abc123")["status"] == "done"
    res = client.get("/api/jobs/abc123/result").json()
    assert res["alphatex"].startswith("\\title") and res["tab"]["bpm"] == 120.0


def test_upload_job_and_pipeline_error(client):
    r = client.post("/api/jobs", files={"file": ("../my song.mp3", b"ID3fake", "audio/mpeg")})
    assert wait(client, r.json()["job_id"])["status"] == "done"
    src = Path(SOURCES[-1])
    assert src.stem == "my_song" and src.parent.name == "upload"   # title source, no traversal
    r = client.post("/api/jobs", data={"url": "https://x.test/fail"})
    s = wait(client, r.json()["job_id"])
    assert s["status"] == "failed" and "라이브" in s["error"]


def test_input_validation(client):
    assert client.post("/api/jobs", data={}).status_code == 400                        # neither
    assert client.post("/api/jobs", data={"url": "ftp://x"}).status_code == 400
    assert client.post("/api/jobs", data={"url": "https://a.b/c", "tuning": "7-string"}).status_code == 400
    assert client.post("/api/jobs", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
    assert client.get("/api/jobs/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
