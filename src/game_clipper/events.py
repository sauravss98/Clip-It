"""Pure functions that turn flagged windows into ranked, clip-ready events.

Nothing here touches ffmpeg or the filesystem, which keeps it easy to test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from game_clipper.detectors.base import Event


@dataclass(frozen=True)
class Selection:
    """Result of event selection: what was kept and what was dropped (and why)."""

    kept: list[Event]
    dropped: list[tuple[Event, str]]


def mask_to_runs(mask: Sequence[bool] | np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index pairs (stop exclusive) for each run of True values."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    edges = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _combine(a: Event, b: Event) -> Event:
    """Merge two events, keeping the stronger peak and weighting means by active time."""
    total = a.active + b.active
    mean = (a.mean * a.active + b.mean * b.active) / total if total else max(a.mean, b.mean)
    stronger = a if a.peak >= b.peak else b
    reasons = tuple(dict.fromkeys(a.reasons + b.reasons))  # de-duplicate, keep order
    return replace(
        stronger,
        start=min(a.start, b.start),
        end=max(a.end, b.end),
        mean=mean,
        transient=max(a.transient, b.transient),
        active=total,
        loud_db=max(a.loud_db, b.loud_db),
        music=min(a.music, b.music),
        reasons=reasons,
    )


def merge_events(events: Sequence[Event], gap: float) -> list[Event]:
    """Merge events whose gap is ``gap`` seconds or less (overlaps always merge)."""
    merged: list[Event] = []
    for event in sorted(events, key=lambda e: e.start):
        if merged and event.start - merged[-1].end <= gap:
            merged[-1] = _combine(merged[-1], event)
        else:
            merged.append(event)
    return merged


def pad_events(
    events: Sequence[Event], before: float, after: float, duration: float
) -> list[Event]:
    """Extend each event by ``before``/``after`` seconds, clamped to ``[0, duration]``."""
    return [
        replace(e, start=max(0.0, e.start - before), end=min(duration, e.end + after))
        for e in events
    ]


def clamp_length(event: Event, min_len: float, max_len: float, duration: float) -> Event:
    """Grow short events around their centre and trim long ones around their peak.

    The result always stays inside ``[0, duration]``; if the whole video is shorter
    than ``min_len`` the event simply spans the video.
    """
    length = event.end - event.start
    if length < min_len:
        target = min(min_len, duration)
        centre = (event.start + event.end) / 2
    elif length > max_len:
        target = max_len
        # Keep the peak in the clip, a little before the middle so the payoff lands.
        centre = min(
            max(event.peak_time + max_len * 0.1, event.start + max_len / 2), event.end - max_len / 2
        )
    else:
        return event

    start = centre - target / 2
    start = min(max(0.0, start), max(0.0, duration - target))
    return replace(event, start=start, end=min(duration, start + target))


def explain(event: Event) -> tuple[str, ...]:
    """Human-readable reasons built from an event's (merged) metrics plus its notes."""
    out = []
    if event.loud_db >= 3.0:
        out.append(f"loud: +{event.loud_db:.1f} dB over baseline")
    if event.transient * 10.0 >= 3.0:
        out.append(f"sharp onset: +{event.transient * 10.0:.1f} dB jump")
    if event.active >= 6.0:
        out.append(f"{event.active:.0f}s of sustained action")
    if event.music < 0.95:
        out.append(f"music-like audio penalised (x{event.music:.2f})")
    return tuple(dict.fromkeys(out + list(event.reasons)))


def score_event(event: Event) -> float:
    """Ranking score from 0 to 100.

    Combines the peak window score, the average intensity, the strongest sudden
    jump and (a little) how long the action lasted. The curve saturates, so the
    number is for ranking and file names rather than an absolute measure.
    """
    raw = (
        0.45 * event.peak
        + 0.25 * event.mean
        + 0.20 * event.transient
        + 0.25 * min(event.active / 10.0, 1.0)
    )
    return round(100.0 * (1.0 - math.exp(-raw / 1.2)), 1)


def select_events(
    candidates: Sequence[Event],
    *,
    duration: float,
    merge_gap: float,
    pad_before: float,
    pad_after: float,
    min_len: float,
    max_len: float,
    top: int,
) -> Selection:
    """Merge, pad, clamp, score and rank candidates; keep the top N in time order."""
    events = merge_events(candidates, merge_gap)
    events = pad_events(events, pad_before, pad_after, duration)
    # Padding can make neighbours overlap; overlapping clips would duplicate footage.
    events = merge_events(events, 0.0)
    events = [clamp_length(e, min_len, max_len, duration) for e in events]
    events = [replace(e, score=score_event(e), reasons=explain(e)) for e in events]

    ranked = sorted(events, key=lambda e: e.score, reverse=True)
    kept = sorted(ranked[:top], key=lambda e: e.start)
    dropped = [
        (e, f"ranked #{i} (only the top {top} are kept)")
        for i, e in enumerate(ranked[top:], start=top + 1)
    ]
    return Selection(kept=kept, dropped=dropped)


def format_hms(seconds: float) -> str:
    """Format seconds as ``01h23m45s`` (used in clip file names)."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}h{total % 3600 // 60:02d}m{total % 60:02d}s"


def format_clock(seconds: float) -> str:
    """Format seconds as ``1:23:45`` or ``3:07`` for display."""
    total = max(0, round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
