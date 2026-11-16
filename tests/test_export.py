from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import requires_ffmpeg
from game_clipper import export, ffmpeg, report
from game_clipper.detectors.base import Event


def event(start: float, end: float, score: float = 87.0) -> Event:
    return Event(
        start=start,
        end=end,
        peak_time=(start + end) / 2,
        score=score,
        reasons=("loud: +12.0 dB over baseline",),
    )


def test_clip_filename() -> None:
    assert export.clip_filename(12, event(5025.0, 5040.0, 87.4)) == "0012_01h23m45s_score87.mp4"


def test_build_command_reencode_keeps_paths_with_spaces_intact() -> None:
    src = Path("C:/Users/Some One/Videos/my game.mkv")
    cmd = export.build_cut_command(
        src, Path("out dir/clip.mp4"), 10, 25, export.ExportOptions(), "h264_nvenc"
    )
    assert str(src) in cmd and str(Path("out dir/clip.mp4")) == cmd[-1]
    assert cmd[cmd.index("-ss") + 1] == "10.000" and cmd[cmd.index("-t") + 1] == "15.000"
    assert "h264_nvenc" in cmd


def test_build_command_copy_mode() -> None:
    cmd = export.build_cut_command(
        Path("a.mp4"), Path("b.mp4"), 0, 5, export.ExportOptions(reencode=False)
    )
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "-filter_complex" not in cmd


def test_build_command_vertical_with_tonemap() -> None:
    cmd = export.build_cut_command(
        Path("a.mp4"), Path("b.mp4"), 0, 5, export.ExportOptions(vertical=True, tonemap=True)
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.startswith("[0:v]zscale") and "1080:1920" in graph and "boxblur" in graph


@requires_ffmpeg
def test_cut_clip_duration(sample_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "clip 1.mp4"
    used = export.cut_clip(sample_video, out, event(2.0, 7.0), export.ExportOptions())
    assert used in {"h264_nvenc", "libx264"}
    info = ffmpeg.probe(out)
    assert info.duration == pytest.approx(5.0, abs=0.15)
    assert (info.width, info.height) == (320, 180)
    assert info.has_audio


@requires_ffmpeg
def test_cut_clip_libx264_fallback_path(sample_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "x264.mp4"
    used = export.cut_clip(
        sample_video, out, event(1.0, 4.0), export.ExportOptions(), encoder="libx264"
    )
    assert used == "libx264"
    assert ffmpeg.probe(out).duration == pytest.approx(3.0, abs=0.15)


@requires_ffmpeg
def test_vertical_output_is_1080x1920(sample_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "vertical.mp4"
    export.cut_clip(sample_video, out, event(0.0, 2.0), export.ExportOptions(vertical=True))
    info = ffmpeg.probe(out)
    assert (info.width, info.height) == (1080, 1920)


@requires_ffmpeg
def test_copy_mode(sample_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "copy.mp4"
    assert (
        export.cut_clip(sample_video, out, event(2.0, 8.0), export.ExportOptions(reencode=False))
        == "copy"
    )
    # Stream copy snaps to keyframes, so only a loose duration check makes sense.
    assert 4.0 <= ffmpeg.probe(out).duration <= 10.0


def test_reports(tmp_path: Path) -> None:
    records = [
        report.ClipRecord(1, event(10, 25), "0001_00h00m10s_score87.mp4"),
        report.ClipRecord(2, event(40, 55, 64), "0002_<odd> name.mp4"),
    ]
    video = Path("my video.mkv")
    report.write_events_json(
        tmp_path / "events.json",
        records,
        video=video,
        duration=100,
        preset="r6siege",
        settings={"top": 20},
    )
    data = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert data["preset"] == "r6siege" and len(data["events"]) == 2
    assert data["events"][0]["clip"] == "0001_00h00m10s_score87.mp4"
    assert data["events"][0]["reasons"] == ["loud: +12.0 dB over baseline"]

    page = report.write_index_html(
        tmp_path / "index.html", records, video=video, preset="r6siege"
    ).read_text(encoding="utf-8")
    assert page.count("<video") == 2
    assert "&lt;odd&gt;" in page and "<odd>" not in page
