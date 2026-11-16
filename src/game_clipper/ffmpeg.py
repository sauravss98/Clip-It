"""ffmpeg/ffprobe helpers: tool checks, probing and streaming decoded audio.

Every command is built as an argument list (never a shell string), so paths with
spaces or unusual characters are passed through untouched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

INSTALL_HINTS = """\
game-clipper needs ffmpeg and ffprobe on your PATH.

  Windows:  winget install Gyan.FFmpeg      (then open a new terminal)
  macOS:    brew install ffmpeg
  Linux:    sudo apt install ffmpeg         (or your distro's package manager)
"""

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ (HDR10) and HLG


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or a command fails."""


@dataclass(frozen=True)
class MediaInfo:
    """Basic facts about a media file, as reported by ffprobe."""

    path: Path
    duration: float
    has_video: bool
    has_audio: bool
    audio_streams: int
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    color_transfer: str | None = None

    @property
    def is_hdr(self) -> bool:
        """True when the video uses an HDR transfer function (PQ or HLG)."""
        return (self.color_transfer or "") in HDR_TRANSFERS


def check_tools() -> None:
    """Ensure ffmpeg and ffprobe are available, raising a helpful error otherwise."""
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FFmpegError(f"Missing: {', '.join(missing)}\n\n{INSTALL_HINTS}")


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing output as text. Raises FFmpegError on failure."""
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise FFmpegError(f"{Path(args[0]).name} failed (exit {proc.returncode}):\n{tail}")
    return proc


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate in ("0/0", "0"):
        return None
    num, _, den = rate.partition("/")
    try:
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return value or None


def probe(path: Path) -> MediaInfo:
    """Probe a media file for duration, streams, resolution, frame rate and HDR."""
    if not path.is_file():
        raise FFmpegError(f"File not found: {path}")
    proc = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = [s for s in streams if s.get("codec_type") == "audio"]

    duration = float(data.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        durations = [float(s["duration"]) for s in streams if s.get("duration")]
        duration = max(durations, default=0.0)

    return MediaInfo(
        path=path,
        duration=duration,
        has_video=video is not None,
        has_audio=bool(audio),
        audio_streams=len(audio),
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        # avg_frame_rate is the honest value for variable-frame-rate captures.
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        if video
        else None,
        video_codec=video.get("codec_name") if video else None,
        color_transfer=video.get("color_transfer") if video else None,
    )


def stream_audio(
    path: Path,
    *,
    sample_rate: int,
    chunk_samples: int,
    audio_track: int = 0,
) -> Iterator[np.ndarray]:
    """Decode one audio track to mono float32 PCM and yield it in chunks.

    The audio is streamed from ffmpeg's stdout, so memory use stays constant no
    matter how long the recording is.
    """
    args = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:a:{audio_track}",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None

    # Drain stderr on a thread so a chatty ffmpeg can never block on a full pipe.
    stderr_lines: list[bytes] = []
    drain = threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True)
    drain.start()

    chunk_bytes = chunk_samples * 4
    leftover = b""
    try:
        while True:
            block = proc.stdout.read(chunk_bytes)
            if not block:
                break
            block = leftover + block
            usable = len(block) - len(block) % 4
            leftover = block[usable:]
            if usable:
                yield np.frombuffer(block[:usable], dtype=np.float32)
    finally:
        proc.stdout.close()
        returncode = proc.wait()
        drain.join(timeout=5)

    if returncode != 0:
        tail = b"".join(stderr_lines[-15:]).decode("utf-8", "replace").strip()
        raise FFmpegError(f"ffmpeg could not decode audio from {path} (exit {returncode}):\n{tail}")


@cache
def encoder_works(encoder: str) -> bool:
    """Return True if ffmpeg can actually encode a tiny test clip with ``encoder``.

    Listing an encoder in ``ffmpeg -encoders`` is not enough: h264_nvenc is
    compiled into most Windows builds but fails at runtime without an NVIDIA GPU.
    """
    proc = run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=256x256:d=0.2",
            "-c:v",
            encoder,
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    return proc.returncode == 0
