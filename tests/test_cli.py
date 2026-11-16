from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from conftest import SR, add_burst, quiet_bed, requires_ffmpeg, write_wav
from game_clipper.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def action_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 60 s video whose audio has bursts at 15 s and 40 s."""
    folder = tmp_path_factory.mktemp("cli media")
    rng = np.random.default_rng(7)
    sig = quiet_bed(60, rng)
    add_burst(sig, 15, rng)
    add_burst(sig, 40, rng)
    wav = write_wav(folder / "audio.wav", sig)
    out = folder / "my match.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=60",
            "-i",
            str(wav),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            str(SR),
            "-shortest",
            str(out),
        ],
        check=True,
    )
    return out


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "game-clipper" in result.stdout


@requires_ffmpeg
def test_detect_writes_events_json(action_video: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["detect", str(action_video), "--out", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    data = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert len(data["events"]) == 2


@requires_ffmpeg
def test_clip_exports_and_writes_review_page(action_video: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["clip", str(action_video), "--out", str(tmp_path), "--top", "1"])
    assert result.exit_code == 0, result.stdout
    clips = list(tmp_path.glob("0001_*_score*.mp4"))
    assert len(clips) == 1
    assert (tmp_path / "index.html").is_file()


@requires_ffmpeg
def test_clip_dry_run_writes_nothing(action_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "dry"
    result = runner.invoke(app, ["clip", str(action_video), "--out", str(out), "--dry-run"])
    assert result.exit_code == 0 and "ffmpeg" in result.stdout
    assert not out.exists()


@requires_ffmpeg
def test_tune_prints_timeline(action_video: Path) -> None:
    result = runner.invoke(app, ["tune", str(action_video), "--preset", "r6siege"])
    assert result.exit_code == 0, result.stdout
    assert "Timeline" in result.stdout and "Picked" in result.stdout


def test_vertical_copy_conflict(tmp_path: Path) -> None:
    video = tmp_path / "x.mp4"
    video.write_bytes(b"")
    result = runner.invoke(app, ["clip", str(video), "--vertical", "--copy"])
    assert result.exit_code == 1


@requires_ffmpeg
def test_unknown_preset(action_video: Path) -> None:
    result = runner.invoke(app, ["detect", str(action_video), "--preset", "minecraft"])
    assert result.exit_code == 1
