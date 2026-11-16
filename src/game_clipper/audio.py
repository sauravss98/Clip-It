"""Per-window audio features: loudness, crest factor and spectral flatness.

Audio is processed chunk by chunk as ffmpeg streams it, so a three-hour recording
only ever holds a few seconds of samples in memory. The output is one small row of
numbers per analysis window (0.5 s by default).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from game_clipper import ffmpeg

SILENCE_DB = -100.0
_EPS = 1e-10


@dataclass(frozen=True)
class Features:
    """Feature timeline, one entry per analysis window."""

    window: float
    rms_db: np.ndarray  # loudness in dBFS
    crest_db: np.ndarray  # peak-to-RMS ratio; high for percussive audio
    flatness: np.ndarray  # 0 = pure tone (music-like), 1 = white noise (gunfire-like)

    @property
    def times(self) -> np.ndarray:
        """Start time of each window in seconds."""
        return np.arange(len(self.rms_db)) * self.window

    def __len__(self) -> int:
        return len(self.rms_db)


def window_features(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (rms_db, crest_db, flatness) for a 2-D array of frames (one per row)."""
    frames = frames.astype(np.float64, copy=False)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    peak = np.max(np.abs(frames), axis=1)
    rms_db = np.maximum(20.0 * np.log10(rms + _EPS), SILENCE_DB)
    crest_db = np.where(rms > 1e-6, 20.0 * np.log10((peak + _EPS) / (rms + _EPS)), 0.0)

    # Spectral flatness: geometric mean / arithmetic mean of the power spectrum.
    # Windowed to limit leakage from strong tones; DC bin dropped.
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), axis=1))[:, 1:] ** 2
    spectrum += _EPS
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)
    flatness = np.where(rms > 1e-6, flatness, 0.0)
    return rms_db, crest_db, flatness


def features_from_chunks(
    chunks: Iterable[np.ndarray],
    *,
    sample_rate: int,
    window: float,
    on_progress: Callable[[float], None] | None = None,
) -> Features:
    """Build a feature timeline from an iterable of mono sample chunks.

    ``on_progress`` is called with the number of seconds processed so far.
    """
    win = max(1, round(window * sample_rate))
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    carry = np.zeros(0, dtype=np.float32)
    processed = 0

    for chunk in chunks:
        buf = np.concatenate([carry, chunk]) if carry.size else chunk
        n_full = len(buf) // win
        if n_full:
            parts.append(window_features(buf[: n_full * win].reshape(n_full, win)))
            processed += n_full * win
            if on_progress:
                on_progress(processed / sample_rate)
        carry = buf[n_full * win :]

    # Keep a trailing partial window if it holds at least half a window of audio.
    if carry.size >= win // 2:
        padded = np.zeros(win, dtype=np.float32)
        padded[: carry.size] = carry
        parts.append(window_features(padded[None, :]))

    if not parts:
        empty = np.zeros(0)
        return Features(window=window, rms_db=empty, crest_db=empty, flatness=empty)
    rms_db, crest_db, flatness = (np.concatenate(cols) for cols in zip(*parts, strict=True))
    return Features(window=window, rms_db=rms_db, crest_db=crest_db, flatness=flatness)


def extract_features(
    path: Path,
    *,
    sample_rate: int,
    window: float,
    audio_track: int = 0,
    on_progress: Callable[[float], None] | None = None,
) -> Features:
    """Stream a media file's audio through ffmpeg and compute its feature timeline."""
    win = max(1, round(window * sample_rate))
    chunks = ffmpeg.stream_audio(
        path,
        sample_rate=sample_rate,
        chunk_samples=win * 120,  # ~60 s of audio per read at the defaults
        audio_track=audio_track,
    )
    return features_from_chunks(
        chunks, sample_rate=sample_rate, window=window, on_progress=on_progress
    )
