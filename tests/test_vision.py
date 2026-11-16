"""Vision stage tests. They use a fake CLIP, so they need neither torch nor a GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_ffmpeg
from game_clipper.detectors.base import Event
from game_clipper.vision import model as hm
from game_clipper.vision import rerank, scenes
from game_clipper.vision.frames import PEAK_OFFSETS, frame_times, grab_frames


def event(start: float, score: float = 50.0) -> Event:
    return Event(start=start, end=start + 15, peak_time=start + 9, peak=1.0, score=score)


class FakeEmbedder:
    """Maps every prompt to a one-hot vector; 'images' match a chosen prompt.

    ``plan`` maps an event start time to the prompt its frames should match.
    """

    def __init__(self, plan: dict[float, str]) -> None:
        self.plan = plan
        self.calls: list[float] = []

    def texts(self, prompts: tuple[str, ...]) -> np.ndarray:
        return np.eye(len(prompts), dtype=np.float32)

    def images(self, frames: np.ndarray) -> np.ndarray:
        start = float(frames[0, 0, 0, 0])  # the fake grabber encodes the event start here
        target = scenes.PROMPTS.index(self.plan[start])
        out = np.zeros((len(frames), len(scenes.PROMPTS)), dtype=np.float32)
        out[:, target] = 1.0
        return out


@pytest.fixture
def fake_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    def grab(video: Path, times: list[float]) -> np.ndarray:
        frames = np.zeros((len(times), 4, 4, 3), dtype=np.float32)
        frames[:, 0, 0, 0] = times[0] - PEAK_OFFSETS[0] - 9  # recover event.start
        return frames

    monkeypatch.setattr(rerank, "grab_frames", grab)


def test_frame_times_stay_inside_event() -> None:
    e = Event(start=10, end=20, peak_time=11)
    times = frame_times(e)
    assert len(times) == len(PEAK_OFFSETS)
    assert all(10 <= t < 20 for t in times)


def test_scene_classification_with_one_hot_prompts() -> None:
    prompts = np.eye(len(scenes.PROMPTS))
    lobby = prompts[[scenes.PROMPTS.index(scenes.NOT_GAMEPLAY[1])]]
    fight = prompts[[scenes.PROMPTS.index(scenes.ACTION_GAMEPLAY[0])]]
    walk = prompts[[scenes.PROMPTS.index(scenes.CALM_GAMEPLAY[2])]]
    assert scenes.classify(lobby, prompts).menu[0] > 0.99
    assert scenes.classify(fight, prompts).action[0] > 0.99
    assert scenes.classify(walk, prompts).action[0] < 0.01


def test_rerank_drops_menus_and_keeps_top_n(fake_frames: None) -> None:
    plan = {
        0.0: scenes.NOT_GAMEPLAY[1],  # loud lobby
        100.0: scenes.ACTION_GAMEPLAY[0],
        200.0: scenes.CALM_GAMEPLAY[0],
        300.0: scenes.NOT_GAMEPLAY[3],  # victory screen
    }
    candidates = [event(0, 90), event(100, 60), event(200, 70), event(300, 95)]
    result = rerank.VisionReranker(embedder=FakeEmbedder(plan)).rerank(  # type: ignore[arg-type]
        Path("x.mp4"), candidates, top=1
    )
    assert [e.start for e in result.selection.kept] == [200.0]  # best remaining audio score
    reasons = {e.start: why for e, why in result.selection.dropped}
    assert "menu" in reasons[0.0] and "menu" in reasons[300.0]
    assert "top 1" in reasons[100.0]
    assert len(result.features) == 1


def test_rerank_uses_trained_model(fake_frames: None) -> None:
    plan = {0.0: scenes.ACTION_GAMEPLAY[0], 100.0: scenes.CALM_GAMEPLAY[0]}
    embedder = FakeEmbedder(plan)
    reranker = rerank.VisionReranker(embedder=embedder)  # type: ignore[arg-type]
    seen = reranker.look(Path("x.mp4"), [event(0), event(100)])
    x = np.stack([v.features(e) for v, e in zip(seen, [event(0), event(100)], strict=True)])
    # A model that loves the "action" frames: train on many copies of each.
    model, _ = hm.train_model(
        np.repeat(x, 10, axis=0), np.repeat([1.0, 0.0], 10), ["a"] * 10 + ["b"] * 10,
        np.zeros(20), preset="test",
    )  # fmt: skip
    reranker.model = model
    kept = reranker.rerank(Path("x.mp4"), [event(0, 40), event(100, 80)], top=1).selection.kept
    assert kept[0].start == 0.0
    assert any("your model" in r for r in kept[0].reasons)


def test_logistic_regression_learns_a_simple_rule() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = ((x[:, 0] + 0.5 * x[:, 1]) > 0).astype(float)
    w, b, mean, std = hm.fit_logistic(x, y, l2=0.1)
    pred = 1 / (1 + np.exp(-(((x - mean) / std) @ w + b)))
    assert ((pred > 0.5) == y).mean() > 0.95
    assert abs(w[2]) < abs(w[0])  # the irrelevant feature gets a small weight


def test_roc_auc() -> None:
    y = np.array([0, 0, 1, 1])
    assert hm.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert hm.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert hm.roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_cross_validation_holds_out_whole_videos() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    y = (x[:, 0] > 0).astype(float)
    groups = [f"video{i % 3}" for i in range(60)]
    report = hm.cross_validate(x, y, groups, baseline=rng.random(60))
    assert report.n_groups == 3
    assert report.model_auc > 0.9 > report.baseline_auc


def test_model_save_load_roundtrip(tmp_path: Path) -> None:
    x = np.random.default_rng(2).normal(size=(40, 5))
    y = (x[:, 0] > 0).astype(float)
    model, _ = hm.train_model(x, y, ["v"] * 40, x[:, 1], preset="fortnite")
    loaded = hm.HighlightModel.load(model.save(tmp_path / "m" / "fortnite.npz"))
    np.testing.assert_allclose(loaded.predict(x), model.predict(x))
    assert loaded.meta["preset"] == "fortnite"


def test_model_path_respects_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GAME_CLIPPER_HOME", str(tmp_path))
    assert hm.model_path("league") == tmp_path / "models" / "league.npz"


@requires_ffmpeg
def test_grab_frames_shape(sample_video: Path) -> None:
    frames = grab_frames(sample_video, [1.0, 5.5, 11.0])
    assert frames.shape == (3, 224, 224, 3) and frames.dtype == np.uint8
    assert frames.mean() > 10  # not blank
    # Letterboxed 16:9: the top rows are black padding.
    assert frames[:, :20].mean() < 1
