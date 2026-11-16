from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import requires_ffmpeg
from game_clipper.web.server import ClipperServer, JobRequest, create_server, safe_filename

pytestmark = requires_ffmpeg


@pytest.fixture
def server(tmp_path: Path) -> Iterator[ClipperServer]:
    srv = create_server("127.0.0.1", 0, tmp_path / "out dir")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def request(
    srv: ClipperServer,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port = srv.server_address[:2]
    conn = http.client.HTTPConnection(str(host), port, timeout=30)
    conn.request(method, path, body=body, headers=headers or {})
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res.status, dict(res.getheaders()), data


def post_json(srv: ClipperServer, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    status, _, data = request(
        srv, "POST", path, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    return status, json.loads(data)


def test_safe_filename() -> None:
    assert safe_filename("..\\..\\evil/../my match.mp4") == "my match.mp4"
    assert safe_filename("bad<>:|name?.mkv") == "bad_name_.mkv"
    assert safe_filename("...") == "upload.mp4"


def test_job_request_validation(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ValueError, match="Choose a video"):
        JobRequest.from_json({})
    with pytest.raises(ValueError, match="not found"):
        JobRequest.from_json({"path": str(tmp_path / "missing.mp4")})
    with pytest.raises(ValueError, match="re-encoding"):
        JobRequest.from_json({"path": str(video), "vertical": True, "copy": True})
    req = JobRequest.from_json({"path": f'"{video}"', "top": "5", "threshold": ""})
    assert req.overrides == {"top": 5} and req.path == video.resolve()


def test_index_and_presets(server: ClipperServer) -> None:
    status, _, body = request(server, "GET", "/")
    assert status == 200 and b"game-clipper" in body
    status, _, body = request(server, "GET", "/api/presets")
    names = {p["name"] for p in json.loads(body)["presets"]}
    assert {"generic", "r6siege", "gta6"} <= names


def test_bad_job_returns_400(server: ClipperServer) -> None:
    status, data = post_json(server, "/api/jobs", {"path": "nope.mp4"})
    assert status == 400 and "not found" in data["error"]
    status, data = post_json(server, "/api/jobs", {"path": __file__, "preset": "minecraft"})
    assert status == 400 and "Unknown preset" in data["error"]


def test_upload_then_clip_end_to_end(server: ClipperServer, sample_video: Path) -> None:
    payload = sample_video.read_bytes()
    status, _, body = request(
        server, "POST", "/api/upload", payload, {"X-Filename": "my%20upload.mp4"}
    )
    assert status == 201
    uploaded = Path(json.loads(body)["path"])
    assert uploaded.read_bytes() == payload and uploaded.parent == server.upload_dir

    # The sample is a steady tone with no action, so force a clip with a zero threshold.
    status, job = post_json(
        server, "/api/jobs", {"path": str(uploaded), "threshold": 0, "top": 1, "min_len": 2}
    )
    assert status == 201
    deadline = time.time() + 60
    while job["status"] not in ("done", "error") and time.time() < deadline:
        time.sleep(0.2)
        _, _, body = request(server, "GET", f"/api/jobs/{job['id']}")
        job = json.loads(body)
    assert job["status"] == "done", job
    assert len(job["clips"]) == 1
    assert (Path(job["out_dir"]) / "index.html").is_file()

    url = job["clips"][0]["url"]
    status, headers, body = request(server, "GET", url, headers={"Range": "bytes=0-99"})
    assert status == 206 and len(body) == 100
    assert headers["Content-Range"].startswith("bytes 0-99/")
    status, _, _ = request(server, "GET", f"/clips/{job['id']}/..%2F..%2Fsecret.txt")
    assert status == 404
