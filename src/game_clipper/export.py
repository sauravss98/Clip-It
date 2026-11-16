"""Cutting clips with ffmpeg: frame-accurate re-encode (NVENC or x264) or stream copy."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from game_clipper import ffmpeg
from game_clipper.detectors.base import Event
from game_clipper.events import format_hms

VERTICAL_W, VERTICAL_H = 1080, 1920

# Best-effort HDR (PQ/HLG) to SDR BT.709 conversion. Needs ffmpeg built with zimg.
TONEMAP_FILTER = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


@dataclass(frozen=True)
class ExportOptions:
    """How clips are cut."""

    reencode: bool = True
    vertical: bool = False
    tonemap: bool = False
    audio_track: int = 0


def clip_filename(index: int, event: Event) -> str:
    """Name like ``0012_01h23m45s_score87.mp4`` (index, source timestamp, score)."""
    return f"{index:04d}_{format_hms(event.start)}_score{round(event.score):d}.mp4"


def pick_video_encoder() -> str:
    """Prefer NVIDIA NVENC; fall back to libx264 when it isn't usable."""
    return "h264_nvenc" if ffmpeg.encoder_works("h264_nvenc") else "libx264"


def vertical_filter(prefix: str = "") -> str:
    """Filter graph for 1080x1920: blurred zoomed background, gameplay fitted on top.

    The background is blurred at quarter resolution and scaled back up, which looks
    the same as a full-size blur and is several times faster.
    """
    w, h = VERTICAL_W, VERTICAL_H
    head = f"[0:v]{prefix}," if prefix else "[0:v]"
    return (
        f"{head}split=2[bg][fg];"
        f"[bg]scale={w // 4}:{h // 4}:force_original_aspect_ratio=increase,"
        f"crop={w // 4}:{h // 4},boxblur=10:2,scale={w}:{h}[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p[v]"
    )


def _encoder_args(encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


def build_cut_command(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    options: ExportOptions,
    encoder: str = "libx264",
) -> list[str]:
    """Build the ffmpeg argument list for one clip. Pure; does not run anything."""
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
    ]
    audio_map = ["-map", f"0:a:{options.audio_track}?"]

    if not options.reencode:
        # Stream copy: very fast, but the cut snaps to the nearest earlier keyframe.
        return [
            *cmd,
            "-map",
            "0:v:0",
            *audio_map,
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(dst),
        ]

    prefix = TONEMAP_FILTER if options.tonemap else ""
    if options.vertical:
        video = ["-filter_complex", vertical_filter(prefix), "-map", "[v]"]
    elif prefix:
        video = ["-map", "0:v:0", "-vf", prefix]
    else:
        video = ["-map", "0:v:0", "-pix_fmt", "yuv420p"]
    return [
        *cmd,
        *video,
        *audio_map,
        *_encoder_args(encoder),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(dst),
    ]


def _run_with_progress(cmd: list[str], on_progress: Callable[[float], None] | None) -> None:
    """Run ffmpeg, reporting seconds encoded from ``-progress pipe:1`` output."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None and proc.stderr is not None
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and on_progress and value.isdigit():
            on_progress(int(value) / 1_000_000)
    stderr = proc.stderr.read()
    if proc.wait() != 0:
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        raise ffmpeg.FFmpegError(f"ffmpeg failed while cutting {cmd[-1]}:\n{tail}")


def cut_clip(
    src: Path,
    dst: Path,
    event: Event,
    options: ExportOptions,
    *,
    encoder: str | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> str:
    """Cut one clip. Returns the video encoder used ("copy" for stream copy).

    If NVENC fails at runtime (driver issue, too many sessions, unsupported
    input), the clip is retried with libx264.
    """
    if not options.reencode:
        _run_with_progress(
            build_cut_command(src, dst, event.start, event.end, options), on_progress
        )
        return "copy"
    encoder = encoder or pick_video_encoder()
    try:
        _run_with_progress(
            build_cut_command(src, dst, event.start, event.end, options, encoder), on_progress
        )
    except ffmpeg.FFmpegError:
        if encoder == "libx264":
            raise
        encoder = "libx264"
        _run_with_progress(
            build_cut_command(src, dst, event.start, event.end, options, encoder), on_progress
        )
    return encoder
