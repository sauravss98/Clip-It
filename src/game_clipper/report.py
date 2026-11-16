"""Output files: events.json and a self-contained index.html review page."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from game_clipper import __version__
from game_clipper.detectors.base import Event
from game_clipper.events import format_clock


@dataclass(frozen=True)
class ClipRecord:
    """An event plus the clip file it was exported to (if any)."""

    index: int
    event: Event
    filename: str | None = None


def event_to_dict(record: ClipRecord) -> dict[str, Any]:
    """Serialise one clip record for events.json."""
    e = record.event
    return {
        "index": record.index,
        "start": round(e.start, 3),
        "end": round(e.end, 3),
        "duration": round(e.duration, 3),
        "start_clock": format_clock(e.start),
        "peak_time": round(e.peak_time, 3),
        "score": e.score,
        "metrics": {
            "peak": round(e.peak, 4),
            "mean": round(e.mean, 4),
            "transient": round(e.transient, 4),
            "active_seconds": round(e.active, 2),
        },
        "reasons": list(e.reasons),
        "source": e.source,
        "clip": record.filename,
    }


def write_events_json(
    path: Path,
    records: Sequence[ClipRecord],
    *,
    video: Path,
    duration: float,
    preset: str,
    settings: dict[str, Any],
) -> Path:
    """Write every selected event with times, scores and reasons."""
    payload = {
        "tool": "game-clipper",
        "version": __version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "video": str(video),
        "duration": round(duration, 3),
        "preset": preset,
        "settings": settings,
        "events": [event_to_dict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; --bg:#0f1115; --card:#181b22; --muted:#8b93a7; --accent:#7c5cff; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
         background:var(--bg); color:#e8eaf0; }}
  header {{ padding:24px 16px 8px; max-width:1400px; margin:auto; }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  header p {{ margin:0; color:var(--muted); }}
  main {{ display:grid; gap:16px; padding:16px; max-width:1400px; margin:auto;
         grid-template-columns:repeat(auto-fill, minmax({min_col}px, 1fr)); }}
  article {{ background:var(--card); border-radius:12px; overflow:hidden; }}
  video {{ width:100%; display:block; background:#000; aspect-ratio:{aspect}; }}
  .meta {{ padding:10px 12px 12px; }}
  .row {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
  .score {{ background:var(--accent); color:#fff; border-radius:999px; padding:1px 10px;
           font-weight:600; font-size:13px; }}
  .name {{ font-family:ui-monospace, Consolas, monospace; font-size:12px; color:var(--muted);
          word-break:break-all; }}
  ul {{ margin:6px 0 0; padding-left:18px; color:var(--muted); font-size:13px; }}
</style>
</head>
<body>
<header>
  <h1>{heading}</h1>
  <p>{summary}</p>
</header>
<main>
{cards}
</main>
</body>
</html>
"""

_CARD = """<article>
  <video src="{src}" controls preload="metadata"></video>
  <div class="meta">
    <div class="row"><strong>#{index} &middot; {clock} &ndash; {end_clock}</strong>
      <span class="score">{score}</span></div>
    <div class="name">{name}</div>
    <ul>{reasons}</ul>
  </div>
</article>"""


def write_index_html(
    path: Path,
    records: Sequence[ClipRecord],
    *,
    video: Path,
    preset: str,
    vertical: bool = False,
) -> Path:
    """Write a self-contained review page with a <video> preview per clip."""
    cards = []
    for r in records:
        if not r.filename:
            continue
        e = r.event
        reasons = "".join(f"<li>{html.escape(x)}</li>" for x in e.reasons) or "<li>flagged</li>"
        cards.append(
            _CARD.format(
                src=html.escape(r.filename, quote=True),
                index=r.index,
                clock=format_clock(e.start),
                end_clock=format_clock(e.end),
                score=f"{e.score:.0f}",
                name=html.escape(r.filename),
                reasons=reasons,
            )
        )
    summary = (
        f"{len(cards)} clips &middot; preset <b>{html.escape(preset)}</b> &middot; sorted by time"
    )
    page = _PAGE.format(
        title=html.escape(f"Clips: {video.stem}"),
        heading=html.escape(video.name),
        summary=summary,
        cards="\n".join(cards) or "<p>No clips were exported.</p>",
        min_col=260 if vertical else 380,
        aspect="9 / 16" if vertical else "16 / 9",
    )
    path.write_text(page, encoding="utf-8")
    return path
