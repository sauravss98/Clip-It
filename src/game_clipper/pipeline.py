"""End-to-end steps shared by the CLI and the web UI: analyse, export, report."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from game_clipper import export, ffmpeg, report
from game_clipper.config import Settings
from game_clipper.detectors.base import Event
from game_clipper.detectors.loudness import LoudnessDetector, Timeline, select
from game_clipper.events import Selection

if TYPE_CHECKING:
    from game_clipper.vision.rerank import VisionReranker

ProgressFn = Callable[[float], None]
VISION_POOL_FACTOR = 3  # vision reviews this many times --top candidates
VISION_FEATURES_FILE = "vision.npz"


class PipelineError(RuntimeError):
    """A user-facing problem with the input (no audio, bad track, ...)."""


@dataclass(frozen=True)
class Analysis:
    """Output of the detection step."""

    info: ffmpeg.MediaInfo
    timeline: Timeline
    selection: Selection
    # Model features per kept event (same order), when the vision stage ran.
    features: list[np.ndarray] | None = None


@dataclass
class ExportResult:
    """Output of the export step."""

    records: list[report.ClipRecord]
    encoders: set[str] = field(default_factory=set)
    events_json: Path | None = None
    index_html: Path | None = None


def probe_input(video: Path, audio_track: int = 0) -> ffmpeg.MediaInfo:
    """Check tools, probe ``video`` and make sure it has the audio we need."""
    ffmpeg.check_tools()
    info = ffmpeg.probe(video)
    if not info.has_audio:
        raise PipelineError(
            f"{video.name} has no audio track; game-clipper detects action from game audio."
        )
    if audio_track >= info.audio_streams:
        raise PipelineError(
            f"Audio track {audio_track} requested but the file has {info.audio_streams} track(s)."
        )
    return info


def analyse(
    video: Path,
    settings: Settings,
    info: ffmpeg.MediaInfo,
    *,
    audio_track: int = 0,
    on_progress: ProgressFn | None = None,
    reranker: VisionReranker | None = None,
    on_vision_progress: Callable[[int, int], None] | None = None,
) -> Analysis:
    """Run the audio detector, then (optionally) the vision reranker.

    ``on_progress`` receives seconds of audio processed; ``on_vision_progress``
    receives ``(candidates_done, candidates_total)``.
    """
    detector = LoudnessDetector(audio_track=audio_track, on_progress=on_progress)
    tl = detector.analyze(video, settings)
    # Prefer the container duration so padding can reach the true end of the video.
    duration = max(info.duration, len(tl.score) * tl.window)
    if reranker is None or not info.has_video:
        return Analysis(info=info, timeline=tl, selection=select(tl, settings, duration))

    # Audio proposes a bigger pool; vision decides which of them make the cut.
    pool_settings = replace(settings, top=max(settings.top * VISION_POOL_FACTOR, 30))
    pool = select(tl, pool_settings, duration)
    total = len(pool.kept)
    result = reranker.rerank(
        video,
        pool.kept,
        settings.top,
        on_progress=(lambda done: on_vision_progress(done, total)) if on_vision_progress else None,
    )
    selection = Selection(
        kept=result.selection.kept, dropped=result.selection.dropped + pool.dropped
    )
    return Analysis(info=info, timeline=tl, selection=selection, features=result.features)


def export_options(
    info: ffmpeg.MediaInfo, *, reencode: bool, vertical: bool, audio_track: int
) -> export.ExportOptions:
    """Export options for ``info``; HDR sources are tone-mapped when re-encoding."""
    if vertical and not reencode:
        raise PipelineError("Vertical output needs re-encoding; turn off stream copy.")
    return export.ExportOptions(
        reencode=reencode,
        vertical=vertical,
        tonemap=info.is_hdr and reencode,
        audio_track=audio_track,
    )


def plan_records(events: Sequence[Event]) -> list[report.ClipRecord]:
    """Assign indexes and file names to events (chronological order)."""
    return [report.ClipRecord(i, e, export.clip_filename(i, e)) for i, e in enumerate(events, 1)]


def export_clips(
    video: Path,
    events: Sequence[Event],
    out_dir: Path,
    options: export.ExportOptions,
    *,
    encoder: str | None = None,
    on_progress: ProgressFn | None = None,
    on_clip: Callable[[report.ClipRecord], None] | None = None,
) -> ExportResult:
    """Cut every event into ``out_dir``.

    ``on_progress`` receives the total seconds of clip footage exported so far;
    ``on_clip`` is called after each finished clip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if options.reencode and encoder is None:
        encoder = export.pick_video_encoder()
    result = ExportResult(records=plan_records(events))
    done = 0.0
    for record in result.records:
        assert record.filename is not None

        def clip_progress(seconds: float, base: float = done) -> None:
            if on_progress:
                on_progress(base + seconds)

        result.encoders.add(
            export.cut_clip(
                video,
                out_dir / record.filename,
                record.event,
                options,
                encoder=encoder,
                on_progress=clip_progress,
            )
        )
        done += record.event.duration
        if on_progress:
            on_progress(done)
        if on_clip:
            on_clip(record)
    return result


def write_reports(
    result: ExportResult,
    out_dir: Path,
    *,
    video: Path,
    duration: float,
    preset: str,
    settings: Settings,
    vertical: bool,
    features: Sequence[np.ndarray] | None = None,
) -> ExportResult:
    """Write events.json and index.html next to the clips.

    With ``features`` (from the vision stage) they are saved to ``vision.npz`` so
    clips rated later can be used for training without re-reading the video.
    """
    if features is not None and len(features) == len(result.records):
        np.savez(
            out_dir / VISION_FEATURES_FILE,
            **{
                r.filename or str(r.index): f for r, f in zip(result.records, features, strict=True)
            },
        )
    result.events_json = report.write_events_json(
        out_dir / "events.json",
        result.records,
        video=video,
        duration=duration,
        preset=preset,
        settings=settings.to_dict(),
    )
    result.index_html = report.write_index_html(
        out_dir / "index.html", result.records, video=video, preset=preset, vertical=vertical
    )
    return result
