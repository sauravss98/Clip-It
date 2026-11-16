"""Grab small RGB frames from a video at chosen timestamps with ffmpeg."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from game_clipper.detectors.base import Event
from game_clipper.ffmpeg import FFmpegError

FRAME_SIZE = 224  # CLIP's input resolution

# Seconds relative to an event's peak. Weighted towards the build-up and the hit
# itself: what happens a few seconds later (a shop menu, a respawn screen) says
# little about whether the moment was exciting.
PEAK_OFFSETS = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0)


def frame_times(event: Event, offsets: Sequence[float] = PEAK_OFFSETS) -> list[float]:
    """Timestamps to sample for ``event``, clamped inside the clip."""
    last = max(event.start, event.end - 0.1)
    return [min(max(event.peak_time + o, event.start), last) for o in offsets]


def grab_frames(video: Path, times: Sequence[float], size: int = FRAME_SIZE) -> np.ndarray:
    """Return ``len(times)`` frames as a uint8 array of shape (N, size, size, 3).

    Each timestamp is a separate fast input seek, and the frames are letterboxed
    (aspect ratio kept, black bars) so HUD elements at the edges stay visible.
    One ffmpeg process handles all timestamps.
    """
    if not times:
        return np.zeros((0, size, size, 3), dtype=np.uint8)
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error"]
    for t in times:
        args += ["-ss", f"{max(0.0, t):.3f}", "-i", str(video)]
    fit = (
        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=rgb24"
    )
    parts = [
        f"[{i}:v:0]{fit},trim=end_frame=1,setpts=PTS-STARTPTS[f{i}]" for i in range(len(times))
    ]
    if len(times) > 1:
        graph = ";".join(parts) + ";" + "".join(f"[f{i}]" for i in range(len(times)))
        graph += f"hstack=inputs={len(times)}[out]"
    else:
        graph = parts[0].replace("[f0]", "[out]")
    args += ["-filter_complex", graph, "-map", "[out]", "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]  # fmt: skip
    proc = subprocess.run(args, capture_output=True)
    expected = size * size * 3 * len(times)
    if proc.returncode != 0 or len(proc.stdout) < expected:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        raise FFmpegError(f"Could not read frames from {video.name}: {' '.join(tail)}")
    strip = np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape(
        size, size * len(times), 3
    )
    return np.stack(np.split(strip, len(times), axis=1))
