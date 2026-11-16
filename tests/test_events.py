from __future__ import annotations

import pytest

from game_clipper.detectors.base import Event
from game_clipper.events import (
    clamp_length,
    format_clock,
    format_hms,
    mask_to_runs,
    merge_events,
    pad_events,
    score_event,
    select_events,
)


def ev(start: float, end: float, peak: float = 1.0, **kw: float) -> Event:
    kw.setdefault("active", end - start)
    kw.setdefault("mean", peak / 2)
    return Event(
        start=start, end=end, peak_time=kw.pop("peak_time", (start + end) / 2), peak=peak, **kw
    )


def test_mask_to_runs() -> None:
    assert mask_to_runs([]) == []
    assert mask_to_runs([False, False]) == []
    assert mask_to_runs([True, True, False, True]) == [(0, 2), (3, 4)]
    assert mask_to_runs([False, True, True, True]) == [(1, 4)]


def test_merge_close_events_and_keep_far_apart() -> None:
    merged = merge_events([ev(10, 12), ev(14, 15), ev(30, 31)], gap=3)
    assert [(e.start, e.end) for e in merged] == [(10, 15), (30, 31)]


def test_merge_combines_metrics() -> None:
    a = ev(0, 2, peak=1.0, transient=0.2, reasons=("a",))  # type: ignore[arg-type]
    b = ev(3, 5, peak=2.0, transient=0.5, peak_time=4.0, reasons=("b", "a"))  # type: ignore[arg-type]
    (m,) = merge_events([b, a], gap=1)
    assert (m.start, m.end) == (0, 5)
    assert m.peak == 2.0 and m.peak_time == 4.0
    assert m.transient == 0.5
    assert m.active == 4.0
    assert m.mean == pytest.approx((0.5 * 2 + 1.0 * 2) / 4)
    assert m.reasons == ("a", "b")


def test_merge_respects_gap_boundary() -> None:
    assert len(merge_events([ev(0, 1), ev(4, 5)], gap=3)) == 1
    assert len(merge_events([ev(0, 1), ev(4.01, 5)], gap=3)) == 2


def test_pad_clamps_to_video_bounds() -> None:
    padded = pad_events([ev(2, 4), ev(50, 58)], before=6, after=4, duration=60)
    assert [(e.start, e.end) for e in padded] == [(0, 8), (44, 60)]


def test_clamp_extends_short_events_around_centre() -> None:
    e = clamp_length(ev(20, 22), min_len=8, max_len=60, duration=100)
    assert (e.start, e.end) == (17, 25)


def test_clamp_short_event_near_start_stays_in_bounds() -> None:
    e = clamp_length(ev(0, 2), min_len=8, max_len=60, duration=100)
    assert (e.start, e.end) == (0, 8)
    e = clamp_length(ev(98, 100), min_len=8, max_len=60, duration=100)
    assert (e.start, e.end) == (92, 100)


def test_clamp_short_video() -> None:
    e = clamp_length(ev(1, 2), min_len=8, max_len=60, duration=5)
    assert (e.start, e.end) == (0, 5)


def test_clamp_trims_long_events_and_keeps_peak() -> None:
    e = clamp_length(ev(0, 200, peak_time=150), min_len=8, max_len=60, duration=300)
    assert e.duration == pytest.approx(60)
    assert e.start <= 150 <= e.end
    # Peak sits a bit before the middle so the payoff has room.
    assert 150 - e.start == pytest.approx(24)


def test_clamp_leaves_valid_lengths_alone() -> None:
    original = ev(10, 30)
    assert clamp_length(original, 8, 60, 100) is original


def test_score_is_monotonic_and_bounded() -> None:
    low = score_event(ev(0, 1, peak=0.5))
    high = score_event(ev(0, 1, peak=2.0))
    longer = score_event(ev(0, 10, peak=2.0))
    assert 0 <= low < high < longer <= 100


def test_select_top_n_returns_chronological_order() -> None:
    candidates = [ev(t, t + 1, peak=p) for t, p in [(10, 0.7), (100, 2.0), (200, 1.0), (300, 1.5)]]
    sel = select_events(
        candidates,
        duration=400,
        merge_gap=4,
        pad_before=6,
        pad_after=4,
        min_len=8,
        max_len=60,
        top=2,
    )
    assert [e.peak for e in sel.kept] == [2.0, 1.5]
    assert [e.start for e in sel.kept] == sorted(e.start for e in sel.kept)
    assert len(sel.dropped) == 2
    assert all("top 2" in why for _, why in sel.dropped)


def test_select_merges_overlap_created_by_padding() -> None:
    sel = select_events(
        [ev(20, 21), ev(28, 29)],
        duration=100,
        merge_gap=4,
        pad_before=6,
        pad_after=4,
        min_len=8,
        max_len=60,
        top=20,
    )
    assert len(sel.kept) == 1
    assert (sel.kept[0].start, sel.kept[0].end) == (14, 33)


def test_select_enforces_lengths() -> None:
    sel = select_events(
        [ev(100, 300, peak_time=200)],
        duration=600,
        merge_gap=4,
        pad_before=6,
        pad_after=4,
        min_len=8,
        max_len=60,
        top=20,
    )
    assert sel.kept[0].duration == pytest.approx(60)


def test_select_empty() -> None:
    sel = select_events(
        [], duration=10, merge_gap=4, pad_before=6, pad_after=4, min_len=8, max_len=60, top=5
    )
    assert sel.kept == [] and sel.dropped == []


def test_time_formatting() -> None:
    assert format_hms(5025.7) == "01h23m45s"
    assert format_clock(65) == "1:05"
    assert format_clock(3725) == "1:02:05"
