from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import MUSIC_BURSTS, MUSIC_SPAN, R6_BURSTS, requires_ffmpeg
from game_clipper.audio import Features, features_from_chunks
from game_clipper.config import load_settings
from game_clipper.detectors.base import Detector
from game_clipper.detectors.loudness import (
    LoudnessDetector,
    analyze,
    onset_strength,
    rolling_percentile,
)

TOLERANCE = 1.5  # seconds between a known burst and the detected peak


def overlaps(event_start: float, event_end: float, span: tuple[float, float]) -> bool:
    return event_start < span[1] and event_end > span[0]


# --- pure helpers ---------------------------------------------------------


def test_features_from_chunks_is_chunking_invariant() -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(16000 * 3).astype(np.float32) * 0.1
    whole = features_from_chunks([signal], sample_rate=16000, window=0.5)
    pieces = features_from_chunks(np.array_split(signal, 7), sample_rate=16000, window=0.5)
    assert len(whole) == 6
    np.testing.assert_allclose(whole.rms_db, pieces.rms_db)


def test_flatness_separates_noise_from_tone() -> None:
    t = np.arange(8000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (0.3 * np.random.default_rng(0).standard_normal(8000)).astype(np.float32)
    f_tone = features_from_chunks([tone], sample_rate=16000, window=0.5)
    f_noise = features_from_chunks([noise], sample_rate=16000, window=0.5)
    assert f_tone.flatness[0] < 0.05 < 0.5 < f_noise.flatness[0]
    assert f_noise.crest_db[0] > f_tone.crest_db[0]


def test_rolling_percentile_flat_and_step() -> None:
    np.testing.assert_allclose(rolling_percentile(np.full(50, -30.0), 10, 50), -30.0)
    step = np.r_[np.full(100, -40.0), np.full(100, -10.0)]
    base = rolling_percentile(step, 20, 50)
    assert base[10] == pytest.approx(-40) and base[190] == pytest.approx(-10)


def test_onset_strength_detects_jump_only() -> None:
    db = np.r_[np.full(10, -40.0), np.full(10, -10.0)]
    onset = onset_strength(db, 4)
    assert onset[10] == pytest.approx(30, abs=0.1)
    assert onset[:10].max() == pytest.approx(0)
    assert onset[15:].max() == pytest.approx(0, abs=0.1)


def test_analyze_handles_silence_and_empty() -> None:
    settings = load_settings()
    silent = Features(
        window=0.5, rms_db=np.full(100, -100.0), crest_db=np.zeros(100), flatness=np.zeros(100)
    )
    assert not analyze(silent, settings).flagged.any()
    empty = Features(window=0.5, rms_db=np.zeros(0), crest_db=np.zeros(0), flatness=np.zeros(0))
    assert len(analyze(empty, settings).score) == 0


def test_loudness_detector_satisfies_protocol() -> None:
    assert isinstance(LoudnessDetector(), Detector)


# --- synthetic recordings (need ffmpeg) -----------------------------------


@requires_ffmpeg
@pytest.mark.parametrize("preset", ["generic", "r6siege", "fortnite"])
def test_finds_every_burst_r6_like(r6_audio: Path, preset: str) -> None:
    """(a) Quiet ambience with short loud bursts: every burst is found, nothing else."""
    events = LoudnessDetector().detect(r6_audio, load_settings(preset))
    assert len(events) == len(R6_BURSTS)
    for burst, event in zip(R6_BURSTS, events, strict=True):
        assert abs(event.peak_time - burst) <= TOLERANCE
        assert event.start <= burst <= event.end
        assert any("onset" in r for r in event.reasons)


@requires_ffmpeg
@pytest.mark.parametrize("preset", ["generic", "fortnite", "league", "gta5_online", "gta6"])
def test_bursts_rank_above_steady_music(music_audio: Path, preset: str) -> None:
    """(b) A long steady loud section must rank below short bursts."""
    events = LoudnessDetector().detect(music_audio, load_settings(preset))
    bursts = [e for e in events if any(e.start <= b <= e.end for b in MUSIC_BURSTS)]
    music = [e for e in events if overlaps(e.start, e.end, MUSIC_SPAN)]
    assert len(bursts) == len(MUSIC_BURSTS)
    if music:
        assert min(e.score for e in bursts) > max(e.score for e in music)


@requires_ffmpeg
def test_music_presets_suppress_steady_music(music_audio: Path) -> None:
    events = LoudnessDetector().detect(music_audio, load_settings("gta5_online"))
    assert not [e for e in events if overlaps(e.start, e.end, MUSIC_SPAN)]


@requires_ffmpeg
def test_mixed_content_sanity(mixed_audio: Path) -> None:
    """(c) Fight, single shots, music and an explosion: sensible counts and order."""
    detector = LoudnessDetector()
    events = detector.detect(mixed_audio, load_settings("generic"))
    assert 4 <= len(events) <= 7
    starts = [e.start for e in events]
    assert starts == sorted(starts)
    for moment in (50, 100, 150, 270):  # fight, two shots, explosion
        assert any(e.start <= moment <= e.end for e in events), moment
    top_two = sorted(events, key=lambda e: e.score, reverse=True)[:2]
    assert {round(e.peak_time) // 10 for e in top_two} & {4, 5, 27}  # fight or explosion
    assert all(e.score > 0 for e in events)
    assert all(8 <= e.duration <= 60 for e in events)


@requires_ffmpeg
def test_top_n_limits_results(mixed_audio: Path) -> None:
    detector = LoudnessDetector()
    events = detector.detect(mixed_audio, load_settings("generic", overrides={"top": 2}))
    assert len(events) == 2
    assert detector.selection is not None and len(detector.selection.dropped) >= 2
    assert min(e.score for e in events) >= max(e.score for e, _ in detector.selection.dropped)
