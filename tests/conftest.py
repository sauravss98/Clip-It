"""Shared fixtures: synthetic game-like audio and video generated on the fly.

No media files are committed. Audio is synthesised with numpy (deterministic seeds)
and written as WAV; small videos are generated with ffmpeg's lavfi sources.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

SR = 16000

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def quiet_bed(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Low ambience: a faint hum plus soft noise (walking, looting, menus)."""
    t = np.arange(int(seconds * SR)) / SR
    return 0.01 * np.sin(2 * np.pi * 110 * t) + 0.004 * rng.standard_normal(len(t))


def add_burst(
    signal: np.ndarray,
    at: float,
    rng: np.random.Generator,
    *,
    length: float = 0.6,
    amp: float = 0.5,
) -> None:
    """Add a gunshot/explosion-like burst: broadband noise with a fast decay."""
    n = int(length * SR)
    start = int(at * SR)
    env = np.exp(-np.linspace(0, 5, n))
    signal[start : start + n] += amp * env * rng.standard_normal(n)


def add_music(signal: np.ndarray, start: float, end: float, *, amp: float = 0.12) -> None:
    """Add a steady tonal chord with gentle tremolo (car radio / cutscene music)."""
    i0, i1 = int(start * SR), int(end * SR)
    t = np.arange(i1 - i0) / SR
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63, 440.0))
    tremolo = 1.0 + 0.08 * np.sin(2 * np.pi * 2.0 * t)
    signal[i0:i1] += amp * chord * tremolo / 2


def add_fight(signal: np.ndarray, start: float, end: float, rng: np.random.Generator) -> None:
    """Add a sustained fight: irregular shots over loud broadband noise."""
    i0, i1 = int(start * SR), int(end * SR)
    signal[i0:i1] += 0.08 * rng.standard_normal(i1 - i0)
    for at in np.arange(start, end - 0.5, 0.7) + rng.uniform(
        0, 0.3, len(np.arange(start, end - 0.5, 0.7))
    ):
        add_burst(signal, float(at), rng, length=0.3, amp=0.35)


def write_wav(path: Path, signal: np.ndarray) -> Path:
    """Write mono float audio in [-1, 1] as 16-bit PCM WAV."""
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())
    return path


R6_BURSTS = [20.0, 55.0, 90.0, 125.0]
MUSIC_SPAN = (60.0, 130.0)
MUSIC_BURSTS = [30.0, 160.0, 185.0]


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # A space in the folder name exercises path handling, as on Windows user dirs.
    path = tmp_path_factory.mktemp("synthetic media")
    return path


@pytest.fixture(scope="session")
def r6_audio(media_dir: Path) -> Path:
    """(a) Quiet ambience with a few short loud bursts at known times."""
    rng = np.random.default_rng(1)
    sig = quiet_bed(150, rng)
    for at in R6_BURSTS:
        add_burst(sig, at, rng)
    return write_wav(media_dir / "r6 like.wav", sig)


@pytest.fixture(scope="session")
def music_audio(media_dir: Path) -> Path:
    """(b) A long steady loud music section plus short bursts elsewhere."""
    rng = np.random.default_rng(2)
    sig = quiet_bed(200, rng)
    add_music(sig, *MUSIC_SPAN)
    for at in MUSIC_BURSTS:
        add_burst(sig, at, rng)
    return write_wav(media_dir / "music plus bursts.wav", sig)


@pytest.fixture(scope="session")
def mixed_audio(media_dir: Path) -> Path:
    """(c) Mixed content: a fight, single shots, music and an explosion."""
    rng = np.random.default_rng(3)
    sig = quiet_bed(300, rng)
    add_fight(sig, 40, 60, rng)
    add_burst(sig, 100, rng, amp=0.4)
    add_burst(sig, 150, rng, amp=0.4)
    add_music(sig, 180, 240)
    add_burst(sig, 270, rng, length=1.5, amp=0.8)
    return write_wav(media_dir / "mixed.wav", sig)


@pytest.fixture(scope="session")
def sample_video(media_dir: Path) -> Path:
    """A 12 s 320x180 test video with a tone, for export tests."""
    out = media_dir / "sample video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "60",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
    )
    return out
