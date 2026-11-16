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
SILENCE_GATE_DB = -70.0  # quieter windows are digital silence (OBS start, pauses, loading)


@dataclass(frozen=True)
class Timeline:
    """Per-window detector internals. Used for event building and by ``tune``."""

    window: float
    rms_db: np.ndarray
    smooth_db: np.ndarray  # loudness averaged over a few seconds
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
    view = sliding_window_view(padded, k)
    if np.isnan(coarse).any():
        # NaN marks values to ignore (silence). Windows that are all NaN borrow the
        # nearest valid baseline.
        valid = ~np.isnan(view).all(axis=1)
        coarse_pct = np.full(len(coarse), np.nan)
        if valid.any():
            coarse_pct[valid] = np.nanpercentile(view[valid], percentile, axis=1)
            idx = np.arange(len(coarse))
            coarse_pct = np.interp(idx, idx[valid], coarse_pct[valid])
    else:
        coarse_pct = np.percentile(view, percentile, axis=1)
    if step == 1:
        return coarse_pct
    return np.interp(np.arange(n), np.arange(len(coarse)) * step, coarse_pct)


def smooth_db(rms_db: np.ndarray, size: int) -> np.ndarray:
    """Centred moving average of loudness in the energy domain (returns dB)."""
    if len(rms_db) == 0 or size <= 1:
        return rms_db.copy()
    size = min(size, len(rms_db))
    power = 10.0 ** (rms_db / 10.0)
    half = size // 2
    padded = np.pad(power, (half, size - 1 - half), mode="edge")
    return 10.0 * np.log10(sliding_window_view(padded, size).mean(axis=1) + 1e-12)


def onset_strength(
    rms_db: np.ndarray, lookback: int, floor_db: np.ndarray | None = None
) -> np.ndarray:
    """dB jump of each window over the energy average of the previous ``lookback`` windows.

    With ``floor_db`` (usually the baseline) the reference never drops below the
    recording's normal level, so sound returning after silence is not a "jump".
    """
    if len(rms_db) == 0:
        return rms_db.copy()
    lookback = max(1, lookback)
    power = 10.0 ** (rms_db / 10.0)
    padded = np.concatenate([np.full(lookback, power[0]), power])
    prev_db = 10.0 * np.log10(sliding_window_view(padded[:-1], lookback).mean(axis=1) + 1e-12)
    if floor_db is not None:
        prev_db = np.maximum(prev_db, floor_db)
    return np.maximum(rms_db - prev_db, 0.0)


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
    tonal_seconds: float = 10.0,
    spike_db: float = 6.0,
) -> np.ndarray:
    """Multiplicative factor in ``(0, 1]`` that suppresses music-like audio.

    Two independent cues:

    * **Steadiness**: a long loud stretch with steady loudness looks like radio; its
      windows are penalised unless they spike well above the stretch's typical
      level (a gunshot over the radio).
    * **Tonality**: music is tonal (low spectral flatness), gunfire and explosions
      are noise-like. A ~10 s stretch that is much more tonal than the recording's
      own typical audio (lobby music, victory screens, cutscenes) is penalised by
      up to ``tonal_penalty``. Comparing against the recording itself keeps this
      independent of each game's mix.
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
        factor *= 1.0 - tonal_penalty * relative_tonality(
            flatness, rms_db > SILENCE_GATE_DB, round(tonal_seconds / window)
        )
    return factor


TONAL_RATIO_START = 0.35  # no penalty while local flatness >= 35% of the recording's median
TONAL_RATIO_FULL = 0.12  # full penalty at <= 12%


def relative_tonality(flatness: np.ndarray, audible: np.ndarray, size: int) -> np.ndarray:
    """0..1 per window: how much more tonal the surrounding seconds are than usual."""
    if not audible.any():
        return np.zeros(len(flatness))
    log_flat = np.log10(np.maximum(flatness, 1e-6))
    reference = float(np.median(log_flat[audible]))
    local = rolling_percentile(np.where(audible, log_flat, np.nan), size, 50.0)
    start, full = np.log10(TONAL_RATIO_START), np.log10(TONAL_RATIO_FULL)
    return np.clip((start - (local - reference)) / (start - full), 0.0, 1.0)


def analyze(features: audio.Features, settings: Settings) -> Timeline:
    """Score every window of a feature timeline. Pure function of its inputs."""
    w = features.window
    rms_db = features.rms_db
    audible = rms_db > SILENCE_GATE_DB
    smoothed = smooth_db(rms_db, round(settings.sustain_window / w))
    if audible.any():
        baseline = rolling_percentile(
            np.where(audible, smoothed, np.nan),
            round(settings.baseline_seconds / w),
            settings.baseline_percentile,
        )
    else:
        baseline = np.full(len(rms_db), SILENCE_GATE_DB)
    sustained = np.clip((smoothed - baseline) / DB_PER_UNIT, 0.0, MAX_UNITS)

    onset = onset_strength(rms_db, round(settings.transient_lookback / w), floor_db=baseline)
    transient = np.clip(onset / DB_PER_UNIT, 0.0, MAX_UNITS)
    # A hit lasts a moment: hold the transient for ~1 s so it overlaps the loud part.
    hold = max(1, round(1.0 / w))
    if len(transient) > hold:
        transient = sliding_window_view(np.pad(transient, (hold - 1, 0)), hold).max(axis=1)

    # Music stretches are found on the smoothed loudness that drives the sustained
    # score, so the penalty covers the same windows the smoothing spreads into.
    floor = rolling_percentile(smoothed, round(3 * settings.baseline_seconds / w), 10.0)
    factor = music_factor(
        smoothed,
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
    score = np.where(audible, score, 0.0)
    return Timeline(
        window=w,
        rms_db=rms_db,
        smooth_db=smoothed,
        baseline_db=baseline,
        sustained=sustained,
        transient=transient,
        music_factor=factor,
        score=score,
        threshold=adaptive_threshold(score, audible, settings.threshold, settings.max_flagged),
    )


def adaptive_threshold(
    score: np.ndarray, audible: np.ndarray, threshold: float, max_flagged: float
) -> float:
    """Raise ``threshold`` so at most ``max_flagged`` of the audible windows pass.

    Busy games (Fortnite, League) are loud most of the time, so a fixed threshold
    would flag a third of the video. Capping the flagged fraction keeps only each
    recording's most intense moments. ``max_flagged >= 1`` disables the cap.
    """
    values = score[audible]
    if max_flagged >= 1.0 or len(values) == 0:
        return threshold
    return max(threshold, float(np.quantile(values, 1.0 - max_flagged)))


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
                loud_db=float(np.max(tl.smooth_db[sl] - tl.baseline_db[sl])),
                music=float(np.min(tl.music_factor[sl])),
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
