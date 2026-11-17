"""Collect the user's 👍/👎 ratings into a dataset and train a highlight model.

Ratings live next to the clips (``labels.json`` in each output folder), so the
dataset is simply "every rated clip under the output root". Model features come
from ``vision.npz`` written at export time, or are computed from the source video
if the clips were exported without the vision stage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from game_clipper import report
from game_clipper.detectors.base import Event
from game_clipper.pipeline import VISION_FEATURES_FILE

if TYPE_CHECKING:
    from game_clipper.vision.model import HighlightModel, TrainReport
    from game_clipper.vision.rerank import VisionReranker

LABELS_FILE = "labels.json"
MIN_LABELS = 10


class TrainingError(RuntimeError):
    """Not enough (or unusable) labels to train."""


@dataclass(frozen=True)
class Run:
    """One output folder produced by ``clip``."""

    folder: Path
    video: str
    preset: str
    records: list[report.ClipRecord]
    labels: dict[str, int]


def read_labels(folder: Path) -> dict[str, int]:
    """Ratings for a folder: clip filename -> 1 (good) or 0 (bad)."""
    path = folder / LABELS_FILE
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items() if v in (0, 1)}


def write_label(folder: Path, filename: str, label: int | None) -> dict[str, int]:
    """Set (1/0) or clear (None) the rating of one clip. Returns all labels."""
    labels = read_labels(folder)
    if label is None:
        labels.pop(filename, None)
    elif label in (0, 1):
        labels[filename] = label
    else:
        raise ValueError("label must be 1, 0 or null")
    (folder / LABELS_FILE).write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels


def find_runs(out_root: Path) -> list[Run]:
    """Every output folder under ``out_root`` that has an events.json, newest first."""
    runs = []
    for events_json in out_root.glob("*/events.json"):
        try:
            meta, records = report.read_events_json(events_json)
        except (OSError, ValueError, KeyError):
            continue  # unreadable or foreign file; skip it
        folder = events_json.parent
        runs.append(
            Run(
                folder=folder,
                video=str(meta.get("video", "")),
                preset=str(meta.get("preset", "generic")),
                records=[r for r in records if r.filename],
                labels=read_labels(folder),
            )
        )
    return sorted(runs, key=lambda r: r.folder.stat().st_mtime, reverse=True)


def recording_id(video: str, chunk: int = 1 << 20) -> str:
    """Identify a recording by its content, so copies of one file share an id.

    The UI saves a re-upload of the same video under a new name ("clip (2).mp4"),
    and evaluation must treat both as one recording or clips from it leak between
    training and test folds. Hashing the size plus three 1 MiB samples is instant
    even for multi-GB files. Falls back to the path if the file is gone.
    """
    path = Path(video)
    try:
        size = path.stat().st_size
        digest = hashlib.sha1(str(size).encode())
        with path.open("rb") as f:
            for offset in (0, max(size // 2 - chunk // 2, 0), max(size - chunk, 0)):
                f.seek(offset)
                digest.update(f.read(chunk))
    except OSError:
        return str(path)
    return digest.hexdigest()


@dataclass
class Dataset:
    """Model inputs for every rated clip."""

    x: np.ndarray
    y: np.ndarray
    groups: list[str]  # recording id per clip, for leave-one-video-out evaluation
    baseline: np.ndarray  # audio-only score, the ranking we must beat


def build_dataset(
    runs: list[Run],
    reranker_factory: Callable[[], VisionReranker],
    on_status: Callable[[str], None] | None = None,
) -> Dataset:
    """Gather features and labels for all rated clips in ``runs``."""
    from game_clipper.vision.model import AUDIO_FEATURES

    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    reranker: VisionReranker | None = None

    for run in runs:
        rated = [r for r in run.records if r.filename in run.labels]
        if not rated:
            continue
        cache_path = run.folder / VISION_FEATURES_FILE
        cache: dict[str, np.ndarray] = {}
        if cache_path.is_file():
            with np.load(cache_path) as data:
                cache = {k: data[k] for k in data.files}
        missing = [r for r in rated if r.filename not in cache]
        if missing:
            source = Path(run.video)
            if not source.is_file():
                if on_status:
                    on_status(f"skipping {len(missing)} clips in {run.folder.name}: "
                              f"source video not found ({source})")  # fmt: skip
                rated = [r for r in rated if r.filename in cache]
            else:
                if on_status:
                    on_status(f"looking at {len(missing)} clips from {source.name}")
                reranker = reranker or reranker_factory()
                events: list[Event] = [r.event for r in missing]
                for record, seen in zip(missing, reranker.look(source, events), strict=True):
                    cache[record.filename or ""] = seen.features(record.event)
                np.savez(cache_path, **cache)
        group = recording_id(run.video)
        for record in rated:
            rows.append(cache[record.filename or ""])
            labels.append(run.labels[record.filename or ""])
            groups.append(group)

    if not rows:
        raise TrainingError("No rated clips found. Rate clips with 👍/👎 in `game-clipper ui`.")
    x = np.stack(rows).astype(np.float64)
    score_col = x.shape[1] - len(AUDIO_FEATURES)  # first audio feature = audio score / 100
    return Dataset(x=x, y=np.array(labels, dtype=float), groups=groups, baseline=x[:, score_col])


def train(
    out_root: Path,
    preset: str,
    reranker_factory: Callable[[], VisionReranker],
    on_status: Callable[[str], None] | None = None,
    include_presets: Sequence[str] = (),
) -> tuple[HighlightModel, TrainReport, Path]:
    """Train and save the model for ``preset`` from rated clips under ``out_root``.

    ``include_presets`` also pulls in ratings from runs made with other presets,
    e.g. recordings of this game that were clipped with ``generic``.
    """
    from game_clipper.vision import model as hm

    wanted = {preset, *include_presets}
    runs = [r for r in find_runs(out_root) if r.preset in wanted and r.labels]
    if not runs:
        raise TrainingError(
            f"No rated {preset} clips under {out_root}. Rate clips with 👍/👎 in "
            "`game-clipper ui` first."
        )
    data = build_dataset(runs, reranker_factory, on_status)
    n_pos = int(data.y.sum())
    if len(data.y) < MIN_LABELS or n_pos == 0 or n_pos == len(data.y):
        raise TrainingError(
            f"Need at least {MIN_LABELS} rated clips with both 👍 and 👎 "
            f"(have {len(data.y)}: {n_pos} 👍, {len(data.y) - n_pos} 👎)."
        )
    model, rep = hm.train_model(data.x, data.y, data.groups, data.baseline, preset=preset)
    path = model.save(hm.model_path(preset))
    return model, rep, path
