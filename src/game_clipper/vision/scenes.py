"""Zero-shot scene understanding with CLIP text prompts (no training needed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Screens that are never highlights, however loud they are.
NOT_GAMEPLAY = (
    "a video game main menu with a play button",
    "a video game lobby with a character standing and menu buttons",
    "a loading screen of a video game",
    "a victory screen with a scoreboard table",
    "a defeat screen with a scoreboard table",
    "a post-match results screen with player statistics",
    "a video game settings menu",
    "a mostly black screen",
)
# Gameplay, from calm to intense. The intense ones double as an "action" signal.
CALM_GAMEPLAY = (
    "third person shooter gameplay with a health bar, minimap and weapon slots",
    "MOBA gameplay with an ability bar and a minimap",
    "a player walking around in a video game",
    "a player looting items in a video game",
    "a top-down view of a quiet game map",
)
ACTION_GAMEPLAY = (
    "intense combat with gunfire in a video game",
    "explosions and fighting in a video game",
    "a team fight with many spell effects in a video game",
    "a player getting eliminated in a shooter game",
)
PROMPTS = NOT_GAMEPLAY + CALM_GAMEPLAY + ACTION_GAMEPLAY
LOGIT_SCALE = 100.0  # CLIP's learned temperature, approximately


@dataclass(frozen=True)
class SceneScores:
    """Per-frame zero-shot probabilities."""

    menu: np.ndarray  # P(not gameplay) per frame
    action: np.ndarray  # P(intense gameplay | gameplay) per frame


def classify(image_emb: np.ndarray, prompt_emb: np.ndarray) -> SceneScores:
    """Softmax over cosine similarities to :data:`PROMPTS` for each frame."""
    logits = LOGIT_SCALE * image_emb @ prompt_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    n_menu, n_calm = len(NOT_GAMEPLAY), len(CALM_GAMEPLAY)
    menu = probs[:, :n_menu].sum(axis=1)
    calm = probs[:, n_menu : n_menu + n_calm].sum(axis=1)
    action = probs[:, n_menu + n_calm :].sum(axis=1)
    return SceneScores(menu=menu, action=action / np.maximum(calm + action, 1e-9))
