"""Second stage of the pipeline: look at candidate clips and re-rank them.

Audio proposes a generous pool of candidates; this stage samples a few frames per
candidate, embeds them with CLIP, rejects menus/lobbies/result screens, and then
re-scores what's left, with the user's trained model when one exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from game_clipper.detectors.base import Event
from game_clipper.events import Selection
from game_clipper.vision import model as hm
from game_clipper.vision import scenes
from game_clipper.vision.embedder import ClipEmbedder
from game_clipper.vision.frames import frame_times, grab_frames

MENU_FRAME = 0.6  # a frame "is a menu" above this probability
MENU_REJECT = 0.5  # reject an event when at least this share of its frames are menus
MODEL_WEIGHT = 0.6  # a fully trusted model sets 60% of the score (see HighlightModel.trust)


@dataclass(frozen=True)
class EventVision:
    """What the vision stage saw for one event."""

    image_mean: np.ndarray  # mean CLIP embedding of its frames
    menu: float  # share of frames that look like menus / non-gameplay
    action: float  # mean zero-shot "intense gameplay" probability

    def features(self, event: Event) -> np.ndarray:
        """Feature vector for the trainable model."""
        return hm.feature_vector(event, self.image_mean, self.menu, self.action)


@dataclass
class RerankResult:
    """Re-ranked selection plus model features for each kept event (same order)."""

    selection: Selection
    features: list[np.ndarray]


class VisionReranker:
    """Scores events by looking at them. Loads CLIP lazily on first use."""

    def __init__(
        self,
        embedder: ClipEmbedder | None = None,
        model: hm.HighlightModel | None = None,
        workers: int = 4,
    ) -> None:
        self.embedder = embedder or ClipEmbedder()
        self.model = model
        self.workers = workers
        self._prompts: np.ndarray | None = None

    @classmethod
    def for_preset(cls, preset: str) -> VisionReranker:
        """Reranker that uses the trained model for ``preset`` if one exists."""
        path = hm.model_path(preset)
        model = hm.HighlightModel.load(path) if path.is_file() else None
        return cls(model=model)

    def look(
        self,
        video: Path,
        events: Sequence[Event],
        on_progress: Callable[[int], None] | None = None,
    ) -> list[EventVision]:
        """Sample frames for every event and summarise what CLIP sees."""
        if self._prompts is None:
            self._prompts = self.embedder.texts(scenes.PROMPTS)

        def grab(event: Event) -> np.ndarray:
            return grab_frames(video, frame_times(event))

        results: list[EventVision] = []
        # ffmpeg seeks run in parallel; CLIP runs on the main thread (GPU).
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for i, frames in enumerate(pool.map(grab, events), start=1):
                emb = self.embedder.images(frames)
                s = scenes.classify(emb, self._prompts)
                mean = emb.mean(axis=0)
                results.append(
                    EventVision(
                        image_mean=mean / (np.linalg.norm(mean) + 1e-9),
                        menu=float((s.menu > MENU_FRAME).mean()),
                        action=float(s.action.mean()),
                    )
                )
                if on_progress:
                    on_progress(i)
        return results

    def rerank(
        self,
        video: Path,
        candidates: Sequence[Event],
        top: int,
        on_progress: Callable[[int], None] | None = None,
    ) -> RerankResult:
        """Reject non-gameplay, re-score with the model (if any) and keep the top N."""
        seen = self.look(video, candidates, on_progress)
        dropped: list[tuple[Event, str]] = []
        scored: list[tuple[Event, np.ndarray]] = []
        for event, vis in zip(candidates, seen, strict=True):
            # On-screen text already proved it's a highlight (a VICTORY banner can
            # look like a menu to CLIP), so only audio-only candidates can be rejected.
            if vis.menu >= MENU_REJECT and event.boost <= 0:
                dropped.append((event, f"vision: looks like a menu/lobby/results screen "
                                       f"({vis.menu:.0%} of frames)"))  # fmt: skip
                continue
            features = vis.features(event)
            notes = [f"vision: gameplay, action {vis.action:.0%}"]
            score = event.score
            if self.model is not None:
                p = float(self.model.predict(features)[0])
                weight = MODEL_WEIGHT * self.model.trust
                score = round((1 - weight) * event.score + weight * 100 * p, 1)
                score = max(score, event.boost)
                notes.append(f"your model: {p:.0%} highlight (trust {self.model.trust:.0%})")
            reasons = tuple(dict.fromkeys(event.reasons + tuple(notes)))
            scored.append((replace(event, score=score, reasons=reasons), features))

        scored.sort(key=lambda item: item[0].score, reverse=True)
        kept = sorted(scored[:top], key=lambda item: item[0].start)
        dropped += [(e, f"ranked #{i} after vision (only the top {top} are kept)")
                    for i, (e, _) in enumerate(scored[top:], start=top + 1)]  # fmt: skip
        return RerankResult(
            selection=Selection(kept=[e for e, _ in kept], dropped=dropped),
            features=[f for _, f in kept],
        )
