"""Detector interface and the Event type shared by all detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from game_clipper.config import Settings


@dataclass(frozen=True)
class Event:
    """A candidate highlight.

    ``start``/``end`` are the clip boundaries (after padding). The metric fields
    describe the flagged core of the event and feed the ranking score.
    """

    start: float
    end: float
    peak_time: float
    peak: float = 0.0  # highest window score
    mean: float = 0.0  # mean window score over the flagged core
    transient: float = 0.0  # strongest sudden jump
    active: float = 0.0  # seconds of flagged audio
    loud_db: float = 0.0  # strongest sustained loudness above the baseline (dB)
    music: float = 1.0  # strongest music penalty applied (1 = none)
    score: float = 0.0  # 0-100 ranking score
    reasons: tuple[str, ...] = field(default_factory=tuple)  # detector-specific notes
    source: str = "loudness"

    @property
    def duration(self) -> float:
        """Clip length in seconds."""
        return self.end - self.start


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a recording into a list of events.

    Future detectors (League Live Client API, kill-feed OCR, ...) implement the
    same interface so the CLI, export and report code stay unchanged.
    """

    name: str

    def detect(self, video: Path, settings: Settings) -> list[Event]:
        """Return the selected events for ``video`` in chronological order."""
        ...
