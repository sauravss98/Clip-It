"""A small local web server for game-clipper, built on the standard library only.

It binds to 127.0.0.1 by default and is meant for the person running it. Videos
can be uploaded through the browser, or referenced by local path to avoid copying
multi-gigabyte recordings. Jobs (clipping and model training) run one at a time on
a background thread (the GPU is the bottleneck anyway) and the page polls for
progress. Clips can be rated 👍/👎 to build the user's own training set.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from game_clipper import config, export, ffmpeg, pipeline, report, training, vision
from game_clipper.events import format_clock

COPY_CHUNK = 1024 * 1024
RANGE_CAP = 1024 * 1024  # max bytes per Range response (see Handler._send_file)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_filename(name: str) -> str:
    """Reduce an uploaded file name to a safe basename."""
    base = Path(name.replace("\\", "/")).name
    base = _SAFE_NAME.sub("_", base).strip(" .")
    return base or "upload.mp4"


@dataclass
class JobRequest:
    """Parameters for one clipping job, as sent by the page."""

    path: Path
    preset: str = config.DEFAULT_PRESET
    vertical: bool = False
    copy: bool = False
    audio_track: int = 0
    vision: bool = True
    text: bool = True  # read on-screen text when the preset supports it
    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> JobRequest:
        """Validate a JSON body. Raises ValueError with a user-facing message."""
        raw_path = str(data.get("path") or "").strip().strip('"')
        if not raw_path:
            raise ValueError("Choose a video or enter its path.")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        overrides = {}
        for key in ("top", "threshold", "pad_before", "pad_after", "min_len", "max_len"):
            value = data.get(key)
            if value not in (None, ""):
                try:
                    overrides[key] = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"'{key}' must be a number") from None
        if "top" in overrides:
            overrides["top"] = int(overrides["top"])
        player = str(data.get("player_name") or "").strip()
        if player:
            overrides["player_name"] = player[:40]
        request = cls(
            path=path.resolve(),
            preset=str(data.get("preset") or config.DEFAULT_PRESET),
            vertical=bool(data.get("vertical")),
            copy=bool(data.get("copy")),
            audio_track=int(data.get("audio_track") or 0),
            vision=bool(data.get("vision", True)),
            text=bool(data.get("text", True)),
            overrides=overrides,
        )
        if request.vertical and request.copy:
            raise ValueError("Vertical output needs re-encoding; turn off fast cut.")
        config.load_settings(request.preset, overrides=overrides)  # raises ConfigError
        return request


def clip_dict(folder: Path, record: report.ClipRecord, labels: dict[str, int]) -> dict[str, Any]:
    """JSON for one clip card (shared by jobs and the library)."""
    name = record.filename or ""
    return {
        "index": record.index,
        "folder": folder.name,
        "file": name,
        "url": f"/files/{quote(folder.name)}/{quote(name)}",
        "start": format_clock(record.event.start),
        "end": format_clock(record.event.end),
        "duration": round(record.event.duration, 1),
        "score": round(record.event.score),
        "reasons": list(record.event.reasons),
        "label": labels.get(name),
    }


@dataclass
class Job:
    """State of a job; read by the HTTP thread, written by the worker."""

    id: str
    title: str
    out_dir: Path
    request: JobRequest | None = None  # clip jobs
    task: Callable[[Job], None] | None = None  # other jobs (training)
    kind: str = "clip"
    status: str = "queued"  # queued | analysing | looking | exporting | training | done | error
    progress: float = 0.0
    message: str = "Waiting in queue"
    error: str | None = None
    encoder: str | None = None
    clips: list[report.ClipRecord] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **changes: Any) -> None:
        """Set fields atomically."""
        with self.lock:
            for key, value in changes.items():
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """JSON snapshot for the page."""
        with self.lock:
            clips = list(self.clips)
            data = {
                "id": self.id,
                "kind": self.kind,
                "video": self.title,
                "status": self.status,
                "progress": round(self.progress, 4),
                "message": self.message,
                "error": self.error,
                "encoder": self.encoder,
                "out_dir": str(self.out_dir),
                "vertical": bool(self.request and self.request.vertical),
            }
        labels = training.read_labels(self.out_dir) if clips else {}
        data["clips"] = [clip_dict(self.out_dir, r, labels) for r in clips]
        return data


class JobManager:
    """Queue of jobs processed one at a time by a daemon worker thread."""

    def __init__(self, out_root: Path) -> None:
        self.out_root = out_root
        self.jobs: dict[str, Job] = {}
        self._embedder: Any = None  # one CLIP model shared by all jobs (slow to load)
        self._queue: queue.Queue[Job] = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="clip-worker", daemon=True)
        self._worker.start()

    def reranker(self, preset: str) -> Any:
        """Vision reranker for ``preset`` (with its trained model), or None if unavailable."""
        if not vision.available():
            return None
        from game_clipper.vision import model as hm
        from game_clipper.vision.embedder import ClipEmbedder
        from game_clipper.vision.rerank import VisionReranker

        if self._embedder is None:
            self._embedder = ClipEmbedder()
        path = hm.model_path(preset)
        model = hm.HighlightModel.load(path) if path.is_file() else None
        return VisionReranker(embedder=self._embedder, model=model)

    def _enqueue(self, job: Job) -> Job:
        self.jobs[job.id] = job
        self._queue.put(job)
        return job

    def submit(self, request: JobRequest) -> Job:
        """Queue a clipping job and return it immediately."""
        suffix = f"_{request.preset}" + ("_vertical" if request.vertical else "")
        return self._enqueue(
            Job(
                id=uuid.uuid4().hex[:12],
                title=request.path.name,
                out_dir=self.out_root / f"{request.path.stem}{suffix}",
                request=request,
            )
        )

    def submit_training(self, preset: str) -> Job:
        """Queue training of the user's model for ``preset``."""
        if preset not in config.available_presets():
            raise ValueError(f"Unknown preset '{preset}'")

        def task(job: Job) -> None:
            job.update(status="training", message="Collecting your rated clips")
            if not vision.available():
                raise RuntimeError(vision.INSTALL_HINT)
            _, rep, path = training.train(
                self.out_root,
                preset,
                reranker_factory=lambda: self.reranker(preset),
                on_status=lambda msg: job.update(message=msg),
            )
            verdict = (
                "beats the audio ranking"
                if rep.model_auc > rep.baseline_auc
                else "doesn't beat the audio ranking yet; rate more clips"
            )
            job.update(
                status="done",
                progress=1.0,
                message=(
                    f"Trained on {rep.n_samples} clips ({rep.n_positive} 👍) from "
                    f"{rep.n_groups} video(s). Held-out AUC: audio {rep.baseline_auc:.2f} → "
                    f"your model {rep.model_auc:.2f}; precision@{rep.k}: "
                    f"{rep.baseline_precision_at_k:.0%} → {rep.model_precision_at_k:.0%}. "
                    f"The model {verdict}. Saved to {path}."
                ),
            )

        return self._enqueue(
            Job(
                id=uuid.uuid4().hex[:12],
                title=f"Train {preset} model",
                out_dir=self.out_root,
                task=task,
                kind="train",
            )
        )

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job.task is not None:
                    job.task(job)
                else:
                    run_job(job, self)
            except Exception as exc:  # report anything to the page instead of dying
                job.update(status="error", error=str(exc), message="Failed")
            finally:
                self._queue.task_done()


def run_job(job: Job, manager: JobManager) -> None:
    """Analyse and export one clipping job, updating its progress along the way."""
    req = job.request
    assert req is not None
    settings = config.load_settings(req.preset, overrides=req.overrides)
    info = pipeline.probe_input(req.path, req.audio_track)
    duration = max(info.duration, 1.0)
    reranker = manager.reranker(req.preset) if req.vision and info.has_video else None
    text_detector = None
    if req.text and settings.text_scan and settings.text_events and info.has_video:
        from game_clipper.detectors import text as text_mod

        if text_mod.available():
            text_detector = text_mod.TextDetector()

    def on_text(seconds: float) -> None:
        job.update(
            status="reading",
            message="Reading kill feed and banners",
            progress=min(seconds / duration, 1.0),
        )

    def on_vision(done: int, total: int) -> None:
        note = " with your model" if reranker is not None and reranker.model else ""
        job.update(
            status="looking",
            message=f"Looking at {total} candidate moments{note}",
            progress=done / max(total, 1),
        )

    job.update(status="analysing", message="Listening for action", progress=0.0)
    analysis = pipeline.analyse(
        req.path,
        settings,
        info,
        audio_track=req.audio_track,
        on_progress=lambda s: job.update(progress=min(s / duration, 1.0)),
        reranker=reranker,
        on_vision_progress=on_vision,
        text_detector=text_detector,
        on_text_progress=on_text,
    )
    events = analysis.selection.kept
    if not events:
        job.update(
            status="done",
            progress=1.0,
            message="No action moments found. Try a lower threshold or another preset.",
        )
        return
    if not info.has_video:
        raise pipeline.PipelineError(f"{req.path.name} has no video stream to cut.")

    options = pipeline.export_options(
        info, reencode=not req.copy, vertical=req.vertical, audio_track=req.audio_track
    )
    encoder = None if req.copy else export.pick_video_encoder()
    total = sum(e.duration for e in events) or 1.0
    job.update(
        status="exporting",
        progress=0.0,
        encoder=encoder or "stream copy",
        message=f"Exporting {len(events)} clips"
        + (" (HDR tone-mapped to SDR)" if options.tonemap else ""),
    )

    def on_clip(record: report.ClipRecord) -> None:
        with job.lock:
            job.clips.append(record)
            job.message = f"Exported {len(job.clips)} of {len(events)} clips"

    result = pipeline.export_clips(
        req.path,
        events,
        job.out_dir,
        options,
        encoder=encoder,
        on_progress=lambda s: job.update(progress=min(s / total, 1.0)),
        on_clip=on_clip,
    )
    pipeline.write_reports(
        result,
        job.out_dir,
        video=req.path,
        duration=info.duration,
        preset=req.preset,
        settings=settings,
        vertical=req.vertical,
        features=analysis.features,
    )
    note = ""
    if encoder == "h264_nvenc" and "libx264" in result.encoders:
        note = " (NVENC failed for some clips; used libx264)"
    job.update(status="done", progress=1.0, message=f"Done: {len(events)} clips{note}")


def open_folder(path: Path) -> None:
    """Open a folder in the system file manager."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class ClipperServer(ThreadingHTTPServer):
    """HTTP server carrying the app state for its handlers."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], out_root: Path) -> None:
        super().__init__(address, Handler)
        self.out_root = out_root
        self.upload_dir = out_root / "_uploads"
        self.manager = JobManager(out_root)

    @property
    def url(self) -> str:
        """Base URL of the running server."""
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class Handler(BaseHTTPRequestHandler):
    """Routes for the page, the JSON API and clip playback."""

    server: ClipperServer
    server_version = "game-clipper"

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the terminal quiet; progress is shown in the browser."""

    # -- helpers -----------------------------------------------------------

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"error": message}, status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def _run_folder(self, name: str) -> Path:
        """Resolve an output folder name, refusing anything outside the output root.

        Doesn't require events.json: it's written after the last clip, and clips
        must be playable (and rateable) while a job is still exporting.
        """
        folder = (self.server.out_root / name).resolve()
        if name in ("", ".", "..") or "/" in name or "\\" in name:
            raise ValueError("Bad folder name")
        if folder.parent != self.server.out_root or not folder.is_dir():
            raise ValueError("Unknown clip folder")
        return folder

    def _library(self) -> dict[str, Any]:
        runs = []
        for run in training.find_runs(self.server.out_root):
            runs.append(
                {
                    "folder": run.folder.name,
                    "video": Path(run.video).name,
                    "preset": run.preset,
                    "rated": len(run.labels),
                    "clips": [clip_dict(run.folder, r, run.labels) for r in run.records],
                }
            )
        presets = sorted({r["preset"] for r in runs})
        models = {}
        if vision.available():
            from game_clipper.vision import model as hm

            for name in config.available_presets():
                path = hm.model_path(name)
                if path.is_file():
                    loaded = hm.HighlightModel.load(path)
                    models[name] = {**loaded.meta, "trust": loaded.trust}
        from game_clipper.detectors import text as text_mod

        text_presets = [
            name
            for name in config.available_presets()
            if config.load_settings(name).text_scan and config.load_settings(name).text_events
        ]
        return {
            "runs": runs,
            "presets": presets,
            "models": models,
            "vision": vision.available(),
            "ocr": text_mod.available(),
            "text_presets": text_presets,
        }

    def _job(self, job_id: str) -> Job | None:
        job = self.server.manager.jobs.get(job_id)
        if job is None:
            self._error("Unknown job", HTTPStatus.NOT_FOUND)
        return job

    # -- routes --------------------------------------------------------------

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        parts = [unquote(p) for p in path.strip("/").split("/") if p]
        if path in ("/", "/index.html"):
            page = resources.files("game_clipper.web").joinpath("static/index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif path == "/api/presets":
            presets = [{"name": n, "description": d} for n, d in config.available_presets().items()]
            self._send_json({"presets": presets, "default": config.DEFAULT_PRESET})
        elif path == "/api/library":
            self._send_json(self._library())
        elif path == "/api/upload/existing":
            query = parse_qs(urlparse(self.path).query)
            name = safe_filename(query.get("name", [""])[0])
            size = int(query.get("size", ["-1"])[0] or -1)
            candidate = self.server.upload_dir / name
            same = candidate.is_file() and candidate.stat().st_size == size
            self._send_json({"path": str(candidate.resolve()) if same else None})
        elif len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            if job := self._job(parts[2]):
                self._send_json(job.to_dict())
        elif len(parts) == 3 and parts[0] == "files":
            try:
                folder = self._run_folder(parts[1])
            except ValueError:
                self._error("Unknown clip", HTTPStatus.NOT_FOUND)
                return
            target = folder / parts[2]
            # Only serve exported clips that sit directly in a run folder.
            if target.suffix.lower() != ".mp4" or target.resolve().parent != folder:
                self._error("Unknown clip", HTTPStatus.NOT_FOUND)
            else:
                self._send_file(target)
        else:
            self._error("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        try:
            if path == "/api/upload":
                self._upload()
            elif path == "/api/jobs":
                request = JobRequest.from_json(self._read_json())
                self._send_json(self.server.manager.submit(request).to_dict(), HTTPStatus.CREATED)
            elif path == "/api/label":
                data = self._read_json()
                folder = self._run_folder(str(data.get("folder", "")))
                name = str(data.get("file", ""))
                if not (folder / name).is_file() or Path(name).name != name:
                    raise ValueError("Unknown clip")
                label = data.get("label")
                labels = training.write_label(folder, name, None if label is None else int(label))
                self._send_json({"label": labels.get(name), "rated": len(labels)})
            elif path == "/api/train":
                preset = str(self._read_json().get("preset") or config.DEFAULT_PRESET)
                job = self.server.manager.submit_training(preset)
                self._send_json(job.to_dict(), HTTPStatus.CREATED)
            elif len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "open":
                if job := self._job(parts[2]):
                    if not job.out_dir.is_dir():
                        raise ValueError("The output folder doesn't exist yet.")
                    open_folder(job.out_dir)
                    self._send_json({"ok": True})
            else:
                self._error("Not found", HTTPStatus.NOT_FOUND)
        except (ValueError, config.ConfigError, ffmpeg.FFmpegError) as exc:
            self._error(str(exc))

    def _upload(self) -> None:
        """Stream the raw request body to disk (no multipart; the page sends the File)."""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("Empty upload")
        name = safe_filename(unquote(self.headers.get("X-Filename") or "upload.mp4"))
        self.server.upload_dir.mkdir(parents=True, exist_ok=True)
        target = self.server.upload_dir / name
        n = 1
        while target.exists():
            target = self.server.upload_dir / f"{Path(name).stem} ({n}){Path(name).suffix}"
            n += 1
        remaining = length
        try:
            with target.open("wb") as fh:
                while remaining:
                    chunk = self.rfile.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise ValueError("Upload interrupted")
                    fh.write(chunk)
                    remaining -= len(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        self._send_json({"path": str(target.resolve())}, HTTPStatus.CREATED)

    def _send_file(self, path: Path) -> None:
        """Serve a file with HTTP Range support so <video> can seek.

        Range responses are capped at :data:`RANGE_CAP` bytes. Browsers open
        ``bytes=0-`` for every <video>, read the start, then pause *without closing
        the connection*. Sending the whole file would leave each connection stuck,
        and after six (Chrome's per-host limit) every other clip greys out while it
        waits. Short ranges complete and close; the player asks for more as needed.
        """
        if not path.is_file():
            self._error("Not found", HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        status = HTTPStatus.OK
        if match := re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or ""):
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last:
                start = max(0, size - int(last))
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, start + RANGE_CAP - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = fh.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser stopped reading (seek or pause); not an error


def create_server(
    host: str = "127.0.0.1", port: int = 8765, out_root: Path | None = None
) -> ClipperServer:
    """Create (but don't start) the UI server. Use port 0 for a random free port."""
    ffmpeg.check_tools()
    root = (out_root or Path.cwd() / "output").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return ClipperServer((host, port), root)
