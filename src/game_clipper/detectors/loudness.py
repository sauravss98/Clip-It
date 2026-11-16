"""Audio-only highlight detector.

Fights, kills, gunshots and explosions are louder and more percussive than looting,
walking or menus. Per analysis window this detector computes:

* **sustained**: how far loudness sits above a rolling per-video baseline;
* **transient**: how sharply loudness jumps compared with the previous seconds;
* **music factor**: a penalty for long, steady, tonal stretches (car radio, cutscenes).

The weighted, penalised score is thresholded, and flagged windows become events.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from game_clipper import audio, events
from game_clipper.config import Settings
from game_clipper.detectors.base import Event

DB_PER_UNIT = 10.0  # score units: 1.0 == 10 dB
MAX_UNITS = 3.0


@dataclass(frozen=True)
class Timeline:
    """Per-window detector internals. Used for event building and by ``tune``."""

    window: float
    rms_db: np.ndarray
    baseline_db: np.ndarray
    sustained: np.ndarray
    transient: np.ndarray
    music_factor: np.ndarray
    score: np.ndarray
    threshold: float

    @property
    def times(self) -> np.ndarray:
        """Start time of each window in seconds."""
        return np.arange(len(self.score)) * self.window

    @property
    def flagged(self) -> np.ndarray:
        """Boolean mask of windows whose score reaches the threshold."""
        return self.score >= self.threshold


def rolling_percentile(values: np.ndarray, size: int, percentile: float) -> np.ndarray:
    """Centred rolling percentile with edge padding (same length as ``values``).

    Long inputs are decimated before the percentile and interpolated back, which
    keeps a three-hour recording fast without changing the baseline meaningfully.
    """
    n = len(values)
    if n == 0:
        return values.copy()
    size = max(1, min(size, n))
    step = max(1, size // 30)
    coarse = values[::step]
    k = max(1, size // step)
    half = k // 2
    padded = np.pad(coarse, (half, k - 1 - half), mode="edge")
    coarse_pct = np.percentile(sliding_window_view(padded, k), percentile, axis=1)
    if step == 1:
        return coarse_pct
    return np.interp(np.arange(n), np.arange(len(coarse)) * step, coarse_pct)


def onset_strength(rms_db: np.ndarray, lookback: int) -> np.ndarray:
    """dB jump of each window over the energy average of the previous ``lookback`` windows."""
    if len(rms_db) == 0:
        return rms_db.copy()
    lookback = max(1, lookback)
    power = 10.0 ** (rms_db / 10.0)
    padded = np.concatenate([np.full(lookback, power[0]), power])
    prev_mean = sliding_window_view(padded[:-1], lookback).mean(axis=1)
    return np.maximum(rms_db - 10.0 * np.log10(prev_mean + 1e-12), 0.0)


def music_factor(
    rms_db: np.ndarray,
    flatness: np.ndarray,
    *,
    window: float,
    floor_db: np.ndarray,
    loud_margin_db: float,
    min_len: float,
    max_std_db: float,
    penalty: float,
    tonal_penalty: float,
    spike_db: float = 6.0,
) -> np.ndarray:
    """Multiplicative factor in ``(0, 1]`` that suppresses music-like audio.

    A long loud stretch with steady loudness looks like music or radio; its windows
    are penalised unless they spike well above the stretch's typical level (a
    gunshot over the radio). Tonal windows (low spectral flatness) can be penalised
    further via ``tonal_penalty``.
    """
    factor = np.ones(len(rms_db))
    if penalty > 0:
        loud = rms_db > floor_db + loud_margin_db
        for start, stop in events.mask_to_runs(loud):
            length = (stop - start) * window
            if length < min_len:
                continue
            segment = rms_db[start:stop]
            edge = min(len(segment) // 4, round(1.0 / window))  # ignore fade in/out
            core = segment[edge : len(segment) - edge] if edge else segment
            median = float(np.median(core))
            std = float(np.std(core))
            steadiness = np.clip((length - min_len) / min_len, 0.0, 1.0) * np.clip(
                1.0 - std / max_std_db, 0.0, 1.0
            )
            spike = np.clip((segment - median) / spike_db, 0.0, 1.0)
            factor[start:stop] *= 1.0 - penalty * steadiness * (1.0 - spike)
    if tonal_penalty > 0:
        tonality = np.clip(1.0 - flatness / 0.3, 0.0, 1.0)  # flatness >= 0.3 ~ noisy
        factor *= 1.0 - tonal_penalty * tonality
    return factor


def analyze(features: audio.Features, settings: Settings) -> Timeline:
    """Score every window of a feature timeline. Pure function of its inputs."""
    w = features.window
    rms_db = features.rms_db
    baseline = rolling_percentile(
        rms_db, round(settings.baseline_seconds / w), settings.baseline_percentile
    )
    sustained = np.clip((rms_db - baseline) / DB_PER_UNIT, 0.0, MAX_UNITS)

    onset = onset_strength(rms_db, round(settings.transient_lookback / w))
    transient = np.clip(onset / DB_PER_UNIT, 0.0, MAX_UNITS)
    # A hit lasts a moment: hold the transient for ~1 s so it overlaps the loud part.
    hold = max(1, round(1.0 / w))
    if len(transient) > hold:
        transient = sliding_window_view(np.pad(transient, (hold - 1, 0)), hold).max(axis=1)

    floor = rolling_percentile(rms_db, round(3 * settings.baseline_seconds / w), 10.0)
    factor = music_factor(
        rms_db,
        features.flatness,
        window=w,
        floor_db=floor,
        loud_margin_db=settings.music_floor_db,
        min_len=settings.music_min_len,
        max_std_db=settings.music_max_std_db,
        penalty=settings.music_penalty,
        tonal_penalty=settings.tonal_penalty,
    )
    score = (settings.w_sustained * sustained + settings.w_transient * transient) * factor
    return Timeline(
        window=w,
        rms_db=rms_db,
        baseline_db=baseline,
        sustained=sustained,
        transient=transient,
        music_factor=factor,
        score=score,
        threshold=settings.threshold,
    )


def _reasons(tl: Timeline, sl: slice) -> tuple[str, ...]:
    reasons = []
    above = float(np.max(tl.rms_db[sl] - tl.baseline_db[sl]))
    if above >= 3.0:
        reasons.append(f"loud: +{above:.1f} dB over baseline")
    jump = float(np.max(tl.transient[sl])) * DB_PER_UNIT
    if jump >= 3.0:
        reasons.append(f"sharp onset: +{jump:.1f} dB jump")
    min_factor = float(np.min(tl.music_factor[sl]))
    if min_factor < 0.95:
        reasons.append(f"music-like audio penalised (x{min_factor:.2f})")
    return tuple(reasons)


def candidates_from_timeline(tl: Timeline) -> list[Event]:
    """Turn each run of flagged windows into an un-padded candidate event."""
    out = []
    times = tl.times
    for start, stop in events.mask_to_runs(tl.flagged):
        sl = slice(start, stop)
        peak_idx = start + int(np.argmax(tl.score[sl]))
        out.append(
            Event(
                start=float(times[start]),
                end=float(times[stop - 1] + tl.window),
                peak_time=float(times[peak_idx] + tl.window / 2),
                peak=float(tl.score[peak_idx]),
                mean=float(np.mean(tl.score[sl])),
                transient=float(np.max(tl.transient[sl])),
                active=(stop - start) * tl.window,
                reasons=_reasons(tl, sl),
            )
        )
    return out


def select(tl: Timeline, settings: Settings, duration: float) -> events.Selection:
    """Build candidates from a timeline and run merge/pad/clamp/rank on them."""
    return events.select_events(
        candidates_from_timeline(tl),
        duration=duration,
        merge_gap=settings.merge_gap,
        pad_before=settings.pad_before,
        pad_after=settings.pad_after,
        min_len=settings.min_len,
        max_len=settings.max_len,
        top=settings.top,
    )


class LoudnessDetector:
    """Detector based on sustained loudness, sudden onsets and music suppression."""

    name = "loudness"

    def __init__(
        self,
        *,
        audio_track: int = 0,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        self.audio_track = audio_track
        self.on_progress = on_progress
        self.timeline: Timeline | None = None
        self.selection: events.Selection | None = None

    def analyze(self, video: Path, settings: Settings) -> Timeline:
        """Extract features from ``video`` and return the scored timeline."""
        features = audio.extract_features(
            video,
            sample_rate=settings.sample_rate,
            window=settings.window,
            audio_track=self.audio_track,
            on_progress=self.on_progress,
        )
        self.timeline = analyze(features, settings)
        return self.timeline

    def detect(self, video: Path, settings: Settings) -> list[Event]:
        """Return the top events for ``video`` in chronological order.

        The full timeline and the dropped candidates stay available on
        ``self.timeline`` and ``self.selection`` for reporting and tuning.
        """
        tl = self.analyze(video, settings)
        duration = len(tl.score) * tl.window
        self.selection = select(tl, settings, duration)
        return self.selection.kept
