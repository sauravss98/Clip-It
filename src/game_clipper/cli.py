"""Command-line interface: ``game-clipper detect | clip | tune``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from game_clipper import __version__, config, export, ffmpeg, report
from game_clipper.detectors.base import Event
from game_clipper.detectors.loudness import DB_PER_UNIT, LoudnessDetector, Timeline
from game_clipper.events import Selection, format_clock, mask_to_runs

app = typer.Typer(
    name="game-clipper",
    help="Find action moments in no-commentary gaming videos from game audio alone.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err = Console(stderr=True)

# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

VideoArg = Annotated[
    Path,
    typer.Argument(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Recording to analyse (MP4, MKV, MOV, ...).",
    ),
]
OutOpt = Annotated[
    Path | None,
    typer.Option("--out", "-o", help="Output folder. Default: ./output/<video name>."),
]
PresetOpt = Annotated[
    str,
    typer.Option(
        "--preset",
        "-p",
        help="Game preset: generic, fortnite, league, r6siege, gta5_online, gta6.",
    ),
]
ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c", exists=True, dir_okay=False, help="TOML file overriding presets."
    ),
]
TopOpt = Annotated[int | None, typer.Option("--top", "-n", min=1, help="Keep the N best events.")]
ThresholdOpt = Annotated[
    float | None, typer.Option("--threshold", "-t", help="Window score needed to flag action.")
]
PadBeforeOpt = Annotated[float | None, typer.Option("--pad-before", min=0, help="Seconds before.")]
PadAfterOpt = Annotated[float | None, typer.Option("--pad-after", min=0, help="Seconds after.")]
MinLenOpt = Annotated[float | None, typer.Option("--min-len", min=0, help="Minimum clip length.")]
MaxLenOpt = Annotated[float | None, typer.Option("--max-len", min=1, help="Maximum clip length.")]
TrackOpt = Annotated[
    int,
    typer.Option(
        "--audio-track", min=0, help="Audio track to analyse (OBS multi-track: 0 = first)."
    ),
]
DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Show what would happen; write nothing.")
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Print settings and commands.")]


@dataclass
class Run:
    """Everything produced by the analysis step."""

    video: Path
    info: ffmpeg.MediaInfo
    settings: config.Settings
    preset: str
    timeline: Timeline
    selection: Selection
    out_dir: Path


def _fail(message: str) -> None:
    err.print(f"[bold red]Error:[/] {message}")
    raise typer.Exit(1)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"game-clipper {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Find action moments in no-commentary gaming videos from game audio alone."""


def _analyse(
    video: Path,
    *,
    out: Path | None,
    preset: str,
    config_file: Path | None,
    overrides: dict[str, Any],
    audio_track: int,
    verbose: bool,
) -> Run:
    """Check tools, probe the video, resolve settings and run the detector."""
    try:
        ffmpeg.check_tools()
        settings = config.load_settings(preset, config_file, overrides)
        info = ffmpeg.probe(video)
    except (ffmpeg.FFmpegError, config.ConfigError) as exc:
        _fail(str(exc))
    if not info.has_audio:
        _fail(f"{video.name} has no audio track; game-clipper detects action from game audio.")
    if audio_track >= info.audio_streams:
        _fail(
            f"--audio-track {audio_track} requested but the file has {info.audio_streams} track(s)."
        )

    if verbose:
        console.print(f"[dim]Settings ({preset}):[/] {settings.to_dict()}")
    res = f"{info.width}x{info.height}" if info.has_video else "audio only"
    fps = f" @ {info.fps:.2f} fps" if info.fps else ""
    hdr = " [yellow]HDR[/]" if info.is_hdr else ""
    console.print(
        f"[bold]{video.name}[/]  {format_clock(info.duration)}  {res}{fps}{hdr}  "
        f"preset=[cyan]{preset}[/]"
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analysing audio", total=max(info.duration, 1.0))
        detector = LoudnessDetector(
            audio_track=audio_track,
            on_progress=lambda s: progress.update(task, completed=s),
        )
        try:
            detector.detect(video, settings)
        except ffmpeg.FFmpegError as exc:
            _fail(str(exc))

    assert detector.timeline is not None and detector.selection is not None
    return Run(
        video=video,
        info=info,
        settings=settings,
        preset=preset,
        timeline=detector.timeline,
        selection=detector.selection,
        out_dir=out or Path.cwd() / "output" / video.stem,
    )


def _overrides(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _events_table(events: list[Event], title: str) -> Table:
    table = Table(title=title, title_justify="left", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Len", justify="right")
    table.add_column("Score", justify="right", style="bold cyan")
    table.add_column("Why")
    for i, e in enumerate(events, start=1):
        table.add_row(
            str(i),
            format_clock(e.start),
            format_clock(e.end),
            f"{e.duration:.0f}s",
            f"{e.score:.0f}",
            "; ".join(e.reasons),
        )
    return table


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


@app.command()
def detect(
    video: VideoArg,
    out: OutOpt = None,
    preset: PresetOpt = config.DEFAULT_PRESET,
    config_file: ConfigOpt = None,
    top: TopOpt = None,
    threshold: ThresholdOpt = None,
    pad_before: PadBeforeOpt = None,
    pad_after: PadAfterOpt = None,
    min_len: MinLenOpt = None,
    max_len: MaxLenOpt = None,
    audio_track: TrackOpt = 0,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Detect action moments, print them and write [bold]events.json[/]."""
    run = _analyse(
        video,
        out=out,
        preset=preset,
        config_file=config_file,
        audio_track=audio_track,
        verbose=verbose,
        overrides=_overrides(
            top=top,
            threshold=threshold,
            pad_before=pad_before,
            pad_after=pad_after,
            min_len=min_len,
            max_len=max_len,
        ),
    )
    kept = run.selection.kept
    if not kept:
        console.print("[yellow]No action moments found.[/] Try a lower --threshold or `tune`.")
    else:
        console.print(_events_table(kept, f"{len(kept)} events"))
    if dry_run:
        return
    run.out_dir.mkdir(parents=True, exist_ok=True)
    path = report.write_events_json(
        run.out_dir / "events.json",
        [report.ClipRecord(i, e) for i, e in enumerate(kept, start=1)],
        video=video,
        duration=run.info.duration,
        preset=preset,
        settings=run.settings.to_dict(),
    )
    console.print(f"Wrote [green]{path}[/]")


# ---------------------------------------------------------------------------
# clip
# ---------------------------------------------------------------------------


@app.command()
def clip(
    video: VideoArg,
    out: OutOpt = None,
    preset: PresetOpt = config.DEFAULT_PRESET,
    config_file: ConfigOpt = None,
    top: TopOpt = None,
    threshold: ThresholdOpt = None,
    pad_before: PadBeforeOpt = None,
    pad_after: PadAfterOpt = None,
    min_len: MinLenOpt = None,
    max_len: MaxLenOpt = None,
    vertical: Annotated[
        bool, typer.Option("--vertical", help="1080x1920 for Shorts (blurred background).")
    ] = False,
    reencode: Annotated[
        bool,
        typer.Option(
            "--reencode/--copy",
            help="Re-encode for frame-accurate cuts (NVENC, else x264), or stream-copy "
            "(fast, but cuts snap to keyframes).",
        ),
    ] = True,
    audio_track: TrackOpt = 0,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Detect action moments and export them as clips with a review page."""
    if vertical and not reencode:
        _fail("--vertical needs re-encoding; remove --copy.")
    run = _analyse(
        video,
        out=out,
        preset=preset,
        config_file=config_file,
        audio_track=audio_track,
        verbose=verbose,
        overrides=_overrides(
            top=top,
            threshold=threshold,
            pad_before=pad_before,
            pad_after=pad_after,
            min_len=min_len,
            max_len=max_len,
        ),
    )
    kept = run.selection.kept
    if not kept:
        console.print("[yellow]No action moments found.[/] Try a lower --threshold or `tune`.")
        raise typer.Exit()
    if not run.info.has_video:
        _fail(f"{video.name} has no video stream to cut.")

    tonemap = run.info.is_hdr and reencode
    if tonemap:
        console.print("[yellow]HDR source:[/] tone-mapping clips to SDR (best effort).")
    elif run.info.is_hdr:
        console.print("[yellow]HDR source:[/] --copy keeps HDR; some players will look washed out.")

    options = export.ExportOptions(
        reencode=reencode, vertical=vertical, tonemap=tonemap, audio_track=audio_track
    )
    encoder = export.pick_video_encoder() if reencode else None
    records = [
        report.ClipRecord(i, e, export.clip_filename(i, e)) for i, e in enumerate(kept, start=1)
    ]
    console.print(_events_table(kept, f"{len(kept)} clips"))

    if dry_run:
        for r in records:
            cmd = export.build_cut_command(
                video,
                run.out_dir / (r.filename or ""),
                r.event.start,
                r.event.end,
                options,
                encoder or "libx264",
            )
            console.print(" ".join(f'"{a}"' if " " in a else a for a in cmd), soft_wrap=True)
        return

    run.out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Encoder: [cyan]{encoder or 'stream copy'}[/]  ->  {run.out_dir}")
    total = sum(e.duration for e in kept)
    used: set[str] = set()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting", total=total)
        done = 0.0
        for r in records:
            assert r.filename is not None
            progress.update(task, description=f"Exporting {r.index}/{len(records)}")

            def on_progress(seconds: float, base: float = done) -> None:
                progress.update(task, completed=base + seconds)

            try:
                used.add(
                    export.cut_clip(
                        video,
                        run.out_dir / r.filename,
                        r.event,
                        options,
                        encoder=encoder,
                        on_progress=on_progress,
                    )
                )
            except ffmpeg.FFmpegError as exc:
                _fail(str(exc))
            done += r.event.duration
            progress.update(task, completed=done)

    if encoder == "h264_nvenc" and "libx264" in used:
        console.print("[yellow]NVENC failed for some clips; they were encoded with libx264.[/]")
    meta = {"video": video, "preset": preset}
    report.write_events_json(
        run.out_dir / "events.json",
        records,
        duration=run.info.duration,
        settings=run.settings.to_dict(),
        **meta,
    )
    page = report.write_index_html(run.out_dir / "index.html", records, vertical=vertical, **meta)
    console.print(f"[green]Done.[/] {len(records)} clips. Review them in [bold]{page}[/]")


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------


def _bar(value: float, threshold: float, width: int = 20) -> str:
    scale = max(threshold * 2, 1e-9)
    filled = round(min(value / scale, 1.0) * width)
    mark = round(width / 2)
    cells = ["#" if i < filled else ("|" if i == mark else ".") for i in range(width)]
    color = "green" if value >= threshold else "white"
    return f"[{color}]{''.join(cells)}[/]"


def _near_misses(tl: Timeline, limit: int = 8) -> list[tuple[int, str]]:
    """Strongest local peaks that stayed below the threshold, with an explanation."""
    flagged_near = np.zeros(len(tl.score), dtype=bool)
    pad = round(5 / tl.window)
    for start, stop in mask_to_runs(tl.flagged):
        flagged_near[max(0, start - pad) : stop + pad] = True
    candidates = np.flatnonzero(~flagged_near & (tl.score >= 0.5 * tl.threshold))
    picked: list[int] = []
    for i in sorted(candidates, key=lambda i: tl.score[i], reverse=True):
        if all(abs(i - j) > pad for j in picked):
            picked.append(int(i))
        if len(picked) == limit:
            break
    out = []
    for i in sorted(picked):
        above = tl.rms_db[i] - tl.baseline_db[i]
        why = (
            f"score {tl.score[i]:.2f} < threshold {tl.threshold:.2f}; "
            f"+{above:.1f} dB over baseline, onset +{tl.transient[i] * DB_PER_UNIT:.1f} dB"
        )
        if tl.music_factor[i] < 0.95:
            why += f", music penalty x{tl.music_factor[i]:.2f}"
        out.append((i, why))
    return out


def _plot(run: Run, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _fail('Plotting needs matplotlib: pip install "game-clipper[plot]"')
    tl = run.timeline
    minutes = tl.times / 60
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(minutes, tl.rms_db, lw=0.6, label="loudness (dBFS)")
    ax1.plot(minutes, tl.baseline_db, lw=1.2, label="baseline")
    ax1.set_ylabel("dB")
    ax1.legend(loc="upper right")
    ax2.plot(minutes, tl.score, lw=0.6, label="score")
    ax2.plot(minutes, tl.music_factor, lw=0.8, alpha=0.6, label="music factor")
    ax2.axhline(tl.threshold, color="red", ls="--", lw=1, label="threshold")
    for e in run.selection.kept:
        for ax in (ax1, ax2):
            ax.axvspan(e.start / 60, e.end / 60, color="green", alpha=0.15)
    ax2.set_xlabel("minutes")
    ax2.legend(loc="upper right")
    fig.suptitle(f"{run.video.name} (preset {run.preset})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@app.command()
def tune(
    video: VideoArg,
    out: OutOpt = None,
    preset: PresetOpt = config.DEFAULT_PRESET,
    config_file: ConfigOpt = None,
    top: TopOpt = None,
    threshold: ThresholdOpt = None,
    pad_before: PadBeforeOpt = None,
    pad_after: PadAfterOpt = None,
    min_len: MinLenOpt = None,
    max_len: MaxLenOpt = None,
    plot: Annotated[
        Path | None, typer.Option("--plot", help="Save a PNG plot (needs game-clipper[plot]).")
    ] = None,
    bucket: Annotated[
        float | None, typer.Option("--bucket", min=1, help="Seconds per summary row (auto).")
    ] = None,
    audio_track: TrackOpt = 0,
    dry_run: DryRunOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Show loudness, transient and score over time, and why moments were (not) picked."""
    run = _analyse(
        video,
        out=out,
        preset=preset,
        config_file=config_file,
        audio_track=audio_track,
        verbose=verbose,
        overrides=_overrides(
            top=top,
            threshold=threshold,
            pad_before=pad_before,
            pad_after=pad_after,
            min_len=min_len,
            max_len=max_len,
        ),
    )
    tl = run.timeline
    n = len(tl.score)
    if n == 0:
        _fail("No audio could be decoded.")

    duration = n * tl.window
    size = bucket or max(10.0, round(duration / 40 / 10) * 10)
    per = max(1, round(size / tl.window))
    kept_mask = np.zeros(n, dtype=bool)
    for e in run.selection.kept:
        kept_mask[int(e.start / tl.window) : int(np.ceil(e.end / tl.window))] = True

    table = Table(
        title=f"Timeline ({size:.0f}s per row, threshold {tl.threshold:.2f})",
        title_justify="left",
        header_style="bold",
    )
    for col, justify in [
        ("Time", "right"),
        ("Loud dB", "right"),
        ("+Base", "right"),
        ("Onset", "right"),
        ("Music", "right"),
        ("Score", "right"),
        ("", "left"),
        ("Clip", "center"),
    ]:
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    for i in range(0, n, per):
        sl = slice(i, i + per)
        best = float(tl.score[sl].max())
        table.add_row(
            format_clock(i * tl.window),
            f"{tl.rms_db[sl].max():.1f}",
            f"{(tl.rms_db[sl] - tl.baseline_db[sl]).max():+.1f}",
            f"{tl.transient[sl].max() * DB_PER_UNIT:.1f}",
            f"x{tl.music_factor[sl].min():.2f}",
            f"{best:.2f}",
            _bar(best, tl.threshold),
            "[green]*[/]" if kept_mask[sl].any() else "",
        )
    console.print(table)

    flagged_pct = 100 * tl.flagged.mean()
    console.print(
        f"Baseline median {np.median(tl.baseline_db):.1f} dBFS; "
        f"{flagged_pct:.1f}% of windows above threshold; "
        f"{len(run.selection.kept)} kept, {len(run.selection.dropped)} dropped by --top."
    )
    if run.selection.kept:
        console.print(_events_table(run.selection.kept, "Picked"))
    if run.selection.dropped:
        console.print("[bold]Dropped[/]")
        for e, why in run.selection.dropped[:10]:
            console.print(f"  {format_clock(e.start)}  score {e.score:.0f}: {why}")
    misses = _near_misses(tl)
    if misses:
        console.print("[bold]Near misses[/] (lower --threshold to include these)")
        for i, why in misses:
            console.print(f"  {format_clock(i * tl.window)}  {why}")
    if flagged_pct > 20:
        console.print("[yellow]Tip:[/] lots of audio is flagged; raise --threshold.")
    elif not run.selection.kept:
        console.print("[yellow]Tip:[/] nothing flagged; lower --threshold.")

    if plot and not dry_run:
        _plot(run, plot)
        console.print(f"Saved plot to [green]{plot}[/]")


if __name__ == "__main__":
    app()
