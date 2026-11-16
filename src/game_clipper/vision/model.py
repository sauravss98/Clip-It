"""A small highlight model trained on the user's own 👍/👎 labels.

It's a regularised logistic regression (a "linear probe") on top of frozen CLIP
embeddings plus the audio detector's metrics. Pure numpy: no torch needed to train
or apply it, so training takes seconds even on a CPU.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from game_clipper.detectors.base import Event

FEATURE_VERSION = 1
AUDIO_FEATURES = ("score", "peak", "mean", "transient", "active", "loud_db", "music")


def data_home() -> Path:
    """Per-user folder for trained models (override with GAME_CLIPPER_HOME)."""
    if env := os.environ.get("GAME_CLIPPER_HOME"):
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "game-clipper"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "game-clipper"


def model_path(preset: str) -> Path:
    """Where the model for ``preset`` is stored."""
    return data_home() / "models" / f"{preset}.npz"


def audio_vector(event: Event) -> np.ndarray:
    """Audio metrics of an event, scaled to roughly unit range."""
    return np.array(
        [
            event.score / 100.0,
            event.peak,
            event.mean,
            event.transient,
            min(event.active / 10.0, 3.0),
            event.loud_db / 10.0,
            event.music,
        ],
        dtype=np.float32,
    )


def feature_vector(event: Event, image_mean: np.ndarray, menu: float, action: float) -> np.ndarray:
    """Model input: [mean CLIP embedding, zero-shot scene scores, audio metrics]."""
    return np.concatenate(
        [image_mean.astype(np.float32), np.array([menu, action], np.float32), audio_vector(event)]
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic(
    x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, steps: int = 3000, lr: float = 0.5
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit L2-regularised logistic regression by gradient descent.

    Features are standardised first. Classes are re-weighted so an unbalanced set
    of labels (say 80% 👎) doesn't push every prediction to the majority class.
    Returns ``(weights, bias, mean, std)``.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    xs = (x - mean) / std
    n, d = xs.shape
    pos = max(float(y.sum()), 1.0)
    neg = max(float(n - y.sum()), 1.0)
    sample_w = np.where(y > 0.5, n / (2 * pos), n / (2 * neg))
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        err = (_sigmoid(xs @ w + b) - y) * sample_w
        w -= lr * (xs.T @ err / n + l2 * w / n)
        b -= lr * float(err.mean())
    return w, b, mean, std


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pos, neg = scores[y > 0.5], scores[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


@dataclass
class HighlightModel:
    """Trained model plus the metadata needed to trust (or distrust) it."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    meta: dict = field(default_factory=dict)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Probability of "good highlight" for each row of ``features``."""
        x = np.atleast_2d(features)
        return _sigmoid(((x - self.mean) / self.std) @ self.weights + self.bias)

    def save(self, path: Path) -> Path:
        """Write the model to an ``.npz`` file (creating folders as needed)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, weights=self.weights, bias=np.array(self.bias), mean=self.mean,
            std=self.std, meta=np.array(json.dumps(self.meta)),
        )  # fmt: skip
        return path

    @classmethod
    def load(cls, path: Path) -> HighlightModel:
        """Read a model written by :meth:`save`."""
        with np.load(path) as data:
            return cls(
                weights=data["weights"],
                bias=float(data["bias"]),
                mean=data["mean"],
                std=data["std"],
                meta=json.loads(str(data["meta"])),
            )


@dataclass(frozen=True)
class TrainReport:
    """Cross-validated quality of a model compared with the audio-only baseline."""

    n_samples: int
    n_positive: int
    n_groups: int
    model_auc: float
    baseline_auc: float
    model_precision_at_k: float
    baseline_precision_at_k: float
    k: int


def _precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    top = np.argsort(-scores)[:k]
    return float(y[top].mean()) if len(top) else float("nan")


def cross_validate(
    x: np.ndarray, y: np.ndarray, groups: Sequence[str], baseline: np.ndarray, *, l2: float = 1.0
) -> TrainReport:
    """Leave-one-video-out evaluation.

    Clips from the same recording look alike (same map, skin, audio mix), so a
    random split would let the model "recognise the video" and look better than it
    is. Holding out whole videos measures what we care about: new recordings.
    """
    groups_arr = np.asarray(groups)
    unique = sorted(set(groups_arr))
    predictions = np.full(len(y), np.nan)
    if len(unique) >= 2:
        for g in unique:
            test = groups_arr == g
            train = ~test
            if len(set(y[train])) < 2:
                continue
            w, b, mean, std = fit_logistic(x[train], y[train], l2=l2)
            predictions[test] = _sigmoid(((x[test] - mean) / std) @ w + b)
    else:  # one video only: fall back to 5-fold, clearly optimistic
        folds = np.arange(len(y)) % 5
        for f in range(5):
            test = folds == f
            if len(set(y[~test])) < 2 or not test.any():
                continue
            w, b, mean, std = fit_logistic(x[~test], y[~test], l2=l2)
            predictions[test] = _sigmoid(((x[test] - mean) / std) @ w + b)
    ok = ~np.isnan(predictions)
    k = max(1, min(10, int(y.sum())))
    return TrainReport(
        n_samples=len(y),
        n_positive=int(y.sum()),
        n_groups=len(unique),
        model_auc=roc_auc(y[ok], predictions[ok]),
        baseline_auc=roc_auc(y[ok], baseline[ok]),
        model_precision_at_k=_precision_at_k(y[ok], predictions[ok], k),
        baseline_precision_at_k=_precision_at_k(y[ok], baseline[ok], k),
        k=k,
    )


def train_model(
    x: np.ndarray, y: np.ndarray, groups: Sequence[str], baseline: np.ndarray, *, preset: str,
    l2: float = 1.0,
) -> tuple[HighlightModel, TrainReport]:  # fmt: skip
    """Cross-validate, then fit the final model on all labels."""
    report = cross_validate(x, y, groups, baseline, l2=l2)
    w, b, mean, std = fit_logistic(x, y, l2=l2)
    meta = {
        "preset": preset,
        "feature_version": FEATURE_VERSION,
        "n_samples": report.n_samples,
        "n_positive": report.n_positive,
        "n_videos": report.n_groups,
        "cv_auc": report.model_auc,
        "baseline_auc": report.baseline_auc,
    }
    return HighlightModel(w, b, mean, std, meta), report
