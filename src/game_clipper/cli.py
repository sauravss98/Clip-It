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

from game_clipper import __version__, config, export, ffmpeg, pipeline, report
from game_clipper.detectors.base import Event
from game_clipper.detectors.loudness import DB_PER_UNIT, Timeline
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
VisionOpt = Annotated[
    bool | None,
    typer.Option(
        "--vision/--no-vision",
        help="Look at candidate clips with a local CLIP model to drop menus and re-rank "
        "(default: on when the [vision] extra is installed).",
    ),
]


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
    features: list[Any] | None = None


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
    vision: bool | None = False,
) -> Run:
    """Check tools, probe the video, resolve settings and run the detectors."""
    try:
        settings = config.load_settings(preset, config_file, overrides)
        info = pipeline.probe_input(video, audio_track)
    except (ffmpeg.FFmpegError, config.ConfigError, pipeline.PipelineError) as exc:
        _fail(str(exc))

    if verbose:
        console.print(f"[dim]Settings ({preset}):[/] {settings.to_dict()}")
    res = f"{info.width}x{info.height}" if info.has_video else "audio only"
    fps = f" @ {info.fps:.2f} fps" if info.fps else ""
    hdr = " [yellow]HDR[/]" if info.is_hdr else ""
    console.print(
        f"[bold]{video.name}[/]  {format_clock(info.duration)}  {res}{fps}{hdr}  "
        f"preset=[cyan]{preset}[/]"
    )
    reranker = _make_reranker(preset, vision, info)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analysing audio", total=max(info.duration, 1.0))
        look = progress.add_task("Looking at candidates", total=None, visible=False)

        def on_vision(done: int, total: int) -> None:
            progress.update(task, visible=False)
            progress.update(look, completed=done, total=total, visible=True)

        try:
            analysis = pipeline.analyse(
                video,
                settings,
                info,
                audio_track=audio_track,
                on_progress=lambda s: progress.update(task, completed=s),
                reranker=reranker,
                on_vision_progress=on_vision,
            )
        except ffmpeg.FFmpegError as exc:
            _fail(str(exc))
        except (OSError, RuntimeError) as exc:  # model download / CUDA problems
            _fail(f"Vision stage failed: {exc}. Run again with --no-vision to skip it.")

    return Run(
        video=video,
        info=info,
        settings=settings,
        preset=preset,
        timeline=analysis.timeline,
        selection=analysis.selection,
        out_dir=out or Path.cwd() / "output" / video.stem,
        features=analysis.features,
    )


def _make_reranker(preset: str, vision: bool | None, info: ffmpeg.MediaInfo) -> Any:
    """Build the vision reranker if requested/available, and say what it will do."""
    from game_clipper import vision as vision_pkg

    if vision is False or not info.has_video:
        return None
    if not vision_pkg.available():
        if vision:
            _fail(vision_pkg.INSTALL_HINT)
        return None

    from game_clipper.vision.rerank import VisionReranker

    reranker = VisionReranker.for_preset(preset)
    if reranker.model is not None:
        meta = reranker.model.meta
        console.print(
            f"Vision: local CLIP + [green]your {preset} model[/] "
            f"({meta.get('n_samples')} rated clips, CV AUC {meta.get('cv_auc', float('nan')):.2f})"
        )
    else:
        console.print(
            "Vision: local CLIP (menus/lobbies are dropped). [dim]Rate clips in "
            "`game-clipper ui` and run `game-clipper train` to rank with your own model.[/]"
        )
    return reranker


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
    vision: VisionOpt = None,
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
        vision=vision,
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
    vision: VisionOpt = None,
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
        vision=vision,
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

    options = pipeline.export_options(
        run.info, reencode=reencode, vertical=vertical, audio_track=audio_track
    )
    if options.tonemap:
        console.print("[yellow]HDR source:[/] tone-mapping clips to SDR (best effort).")
    elif run.info.is_hdr:
        console.print("[yellow]HDR source:[/] --copy keeps HDR; some players will look washed out.")

    encoder = export.pick_video_encoder() if reencode else None
    console.print(_events_table(kept, f"{len(kept)} clips"))

    if dry_run:
        for r in pipeline.plan_records(kept):
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

    console.print(f"Encoder: [cyan]{encoder or 'stream copy'}[/]  ->  {run.out_dir}")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting clips", total=sum(e.duration for e in kept))
        try:
            result = pipeline.export_clips(
                video,
                kept,
                run.out_dir,
                options,
                encoder=encoder,
                on_progress=lambda s: progress.update(task, completed=s),
            )
        except ffmpeg.FFmpegError as exc:
            _fail(str(exc))

    if encoder == "h264_nvenc" and "libx264" in result.encoders:
        console.print("[yellow]NVENC failed for some clips; they were encoded with libx264.[/]")
    pipeline.write_reports(
        result,
        run.out_dir,
        video=video,
        duration=run.info.duration,
        preset=preset,
        settings=run.settings,
        vertical=vertical,
        features=run.features,
    )
    console.print(
        f"[green]Done.[/] {len(result.records)} clips. Review them in [bold]{result.index_html}[/]"
    )


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


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------


@app.command()
def ui(
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Folder for clips. Default: ./output.")
    ] = None,
    host: Annotated[str, typer.Option(help="Interface to bind (keep it local).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535, help="Port (0 = any free port).")] = 8765,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Don't open a browser tab.")
    ] = False,
) -> None:
    """Open a local web page: drop in a recording, get clips back."""
    import webbrowser

    from game_clipper.web import create_server

    try:
        server = create_server(host, port, out)
    except ffmpeg.FFmpegError as exc:
        _fail(str(exc))
    except OSError as exc:
        _fail(f"Could not start the server on {host}:{port}: {exc}. Try --port 0.")
    console.print(f"game-clipper UI running at [bold cyan]{server.url}[/]  (Ctrl+C to stop)")
    console.print(f"Clips are saved under {server.out_root}")
    if not no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    app()
