from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from game_clipper import report, training
from game_clipper.detectors.base import Event
from game_clipper.vision.rerank import EventVision


class FakeReranker:
    """Returns an image embedding that encodes whether the clip is 'good'."""

    def __init__(self) -> None:
        self.looked = 0

    def look(self, video: Path, events: list[Event]) -> list[EventVision]:
        self.looked += len(events)
        out = []
        for e in events:
            good = e.start % 100 < 50  # ground truth hidden in the timestamps
            emb = np.zeros(8, np.float32)
            emb[0 if good else 1] = 1.0
            out.append(EventVision(image_mean=emb, menu=0.0, action=0.5))
        return out


def make_run(root: Path, name: str, video: Path, n: int, preset: str = "fortnite") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    records = []
    for i in range(n):
        start = i * 50.0
        event = Event(start=start, end=start + 12, peak_time=start + 6, peak=1.0, score=50.0)
        name_i = f"{i + 1:04d}_clip.mp4"
        (folder / name_i).write_bytes(b"x")
        records.append(report.ClipRecord(i + 1, event, name_i))
    report.write_events_json(folder / "events.json", records, video=video, duration=n * 50,
                             preset=preset, settings={})  # fmt: skip
    for r in records:
        training.write_label(folder, r.filename or "", int(r.event.start % 100 < 50))
    return folder


def test_labels_roundtrip(tmp_path: Path) -> None:
    assert training.write_label(tmp_path, "a.mp4", 1) == {"a.mp4": 1}
    assert training.write_label(tmp_path, "b.mp4", 0) == {"a.mp4": 1, "b.mp4": 0}
    assert training.write_label(tmp_path, "a.mp4", None) == {"b.mp4": 0}
    with pytest.raises(ValueError):
        training.write_label(tmp_path, "a.mp4", 5)


def test_events_json_roundtrip(tmp_path: Path) -> None:
    e = Event(start=1, end=13, peak_time=7, peak=1.5, mean=0.9, transient=2.0, active=4,
              loud_db=12.5, music=0.8, score=77, reasons=("loud",))  # fmt: skip
    records = [report.ClipRecord(1, e, "c.mp4")]
    report.write_events_json(
        tmp_path / "events.json", records, video=Path("v.mp4"), duration=20, preset="r6siege",
        settings={},
    )  # fmt: skip
    meta, records = report.read_events_json(tmp_path / "events.json")
    assert meta["preset"] == "r6siege"
    assert records[0].event == e and records[0].filename == "c.mp4"


def test_train_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GAME_CLIPPER_HOME", str(tmp_path / "home"))
    out = tmp_path / "output"
    for i in range(3):
        video = tmp_path / f"video{i}.mp4"
        video.write_bytes(b"x")
        make_run(out, f"run{i}", video, n=8)
    make_run(out, "league run", tmp_path / "video0.mp4", n=4, preset="league")

    fake = FakeReranker()
    model, rep, path = training.train(out, "fortnite", reranker_factory=lambda: fake)
    assert fake.looked == 24  # only fortnite clips
    assert rep.n_samples == 24 and rep.n_groups == 3
    assert rep.model_auc > 0.9  # the embedding carries the label
    assert path.is_file() and model.meta["preset"] == "fortnite"
    # Features are cached next to the clips, so training again doesn't re-read videos.
    assert (out / "run0" / "vision.npz").is_file()
    training.train(out, "fortnite", reranker_factory=lambda: fake)
    assert fake.looked == 24


def test_train_needs_enough_labels(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    make_run(tmp_path / "out", "run", video, n=3)
    with pytest.raises(training.TrainingError, match="at least"):
        training.train(tmp_path / "out", "fortnite", reranker_factory=FakeReranker)
    with pytest.raises(training.TrainingError, match="No rated"):
        training.train(tmp_path / "out", "league", reranker_factory=FakeReranker)


def test_find_runs_skips_broken_folders(tmp_path: Path) -> None:
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "events.json").write_text("{not json", encoding="utf-8")
    make_run(tmp_path, "ok", tmp_path / "v.mp4", n=2)
    assert [r.folder.name for r in training.find_runs(tmp_path)] == ["ok"]
    assert json.loads((tmp_path / "ok" / "labels.json").read_text())  # labels written
