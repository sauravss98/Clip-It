"""On-screen text detector tests. OCR is faked, so they need no OCR engine."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from game_clipper.config import ConfigError, TextKeyword, load_settings
from game_clipper.detectors import text as T
from game_clipper.detectors.base import Event
from game_clipper.events import select_events

# Real OCR output from a League VOD, typos included.
REAL_LINES = {
    "mine": "Inf3rn00990 (Senna)has slain Laurucu (KafSal for a triple kill",
    "mine_again": "Inf3rn00990 (Senna) has slain Laurucur (KafSa) fora triple kill!",
    "victim": "X has slain Inf3rno0990 (Senna) for a triple kill!",
    "other": "SHIR@ (Shyyana) has slain Dirk (Ashe) for a triple killl",
    "ace": "Allyteamhasscoredan ace!",
    "victory": "JICTORY",
    "double": "Kuka65 (llaoi)has slain reamats (werad Fate)fora doublekilll",
    "noise": "IicnIoorasiyurhssroruune)nassiainiacepurpieyayncyiora",
}
PLAYER = "Inf3rno0990"


@pytest.fixture
def league() -> object:
    return load_settings("league", overrides={"player_name": PLAYER})


def test_normalize_folds_ocr_noise() -> None:
    assert T.normalize("for a triple killl!") == T.normalize("fora triplekil")
    assert T.normalize("Inf3rn00990") == T.normalize("Inf3rno0990")


def test_find_phrase_exact_and_fuzzy() -> None:
    text = T.normalize(REAL_LINES["victory"])
    assert T.find_phrase(text, T.normalize("victory")) == 0  # "JICTORY": one OCR mistake
    assert T.find_phrase(T.normalize("history"), T.normalize("victory")) == -1
    assert T.find_phrase(T.normalize("VICTORY"), T.normalize("victory")) == 0
    assert (
        T.find_phrase(T.normalize("Ally team has scored an ace"), T.normalize("scored an ace")) > 0
    )
    assert (
        T.find_phrase(T.normalize("a nice place for a space race"), T.normalize("scored an ace"))
        == -1
    )


@pytest.mark.parametrize(
    ("key", "phrase", "mine"),
    [
        ("mine", "triple kill", True),
        ("victim", "triple kill", False),  # a kill *on* you isn't yours
        ("other", "triple kill", False),
        ("ace", "ally team has scored an ace", False),
        ("double", "double kill", False),
    ],
)
def test_match_line(league: object, key: str, phrase: str, mine: bool) -> None:
    hits = T.match_line(REAL_LINES[key], 10.0, league.text_events, PLAYER)  # type: ignore[attr-defined]
    assert [h.keyword.phrase for h in hits] == [phrase]
    assert hits[0].mine is mine


def test_noise_and_unrelated_text_do_not_match(league: object) -> None:
    for line in (REAL_LINES["noise"], "plz finish up they are so bad", "+217", "Dirk"):
        assert T.match_line(line, 0.0, league.text_events) == []  # type: ignore[attr-defined]


def test_grouping_scores_and_bonuses() -> None:
    kills = TextKeyword("triple kill", 90)
    victory = TextKeyword("victory", 88, cooldown=300, team=True)
    hits = [
        T.TextHit(100.0, kills, REAL_LINES["mine"], mine=True),
        T.TextHit(105.0, kills, REAL_LINES["mine_again"], mine=True),  # same line re-read
        T.TextHit(160.0, kills, REAL_LINES["other"]),  # someone else, later
        T.TextHit(230.0, kills, REAL_LINES["other"]),  # stale line still in chat history
        T.TextHit(500.0, victory, "VICTORY"),
        T.TextHit(530.0, victory, "VICTORY"),  # post-game screen: same moment
    ]
    events = T.hits_to_events(hits, PLAYER)
    assert [(e.peak_time, e.boost) for e in events] == [
        (100.0, 100.0),
        (160.0, 72.0),
        (500.0, 88.0),
    ]
    assert "(you)" in events[0].reasons[0]
    assert all(e.source == "text" for e in events)
    assert events[0].start == 100.0 - T.BEFORE


def test_no_player_name_means_no_bonus_or_discount() -> None:
    kills = TextKeyword("triple kill", 90)
    (event,) = T.hits_to_events([T.TextHit(10.0, kills, REAL_LINES["other"])])
    assert event.boost == 90


def test_crop_regions_scale_and_min_height() -> None:
    frame = np.zeros((720, 1280, 3), np.uint8)
    (chat, min0), (banner, min1) = T.crop_regions(
        frame, [(0.0, 0.5, 0.5, 1.0, 1.0, 0.0), (0.25, 0.0, 0.75, 0.5, 0.5, 18.0)]
    )
    assert chat.shape == (360, 640, 3) and min0 == 0
    assert banner.shape == (180, 320, 3) and min1 == 9.0


def test_scan_with_fake_reader(league: object) -> None:
    script = {0.0: ["walking"], 5.0: [REAL_LINES["mine"]], 10.0: [REAL_LINES["mine_again"]],
              15.0: [], 20.0: ["VICTORY"]}  # fmt: skip

    class FakeReader:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self, image: np.ndarray, min_height: float = 0.0) -> list[str]:
            # Only the first region (chat) "contains" the scripted text.
            return script.get(float(image[0, 0, 0]), []) if image.shape[0] > 200 else []

    def frames() -> Iterator[tuple[float, np.ndarray]]:
        for t in script:
            frame = np.zeros((720, 1280, 3), np.uint8)
            frame[:, :, 0] = int(t)  # encode the time for the fake reader
            yield t, frame

    seen: list[float] = []
    detector = T.TextDetector(reader_factory=FakeReader, on_progress=seen.append)
    events = detector.scan(frames(), league)  # type: ignore[arg-type]
    assert [(e.peak_time, e.boost) for e in events] == [(5.0, 100.0), (20.0, 88.0)]
    assert seen == list(script)


def test_text_events_outrank_audio_and_set_peak() -> None:
    audio = Event(start=50, end=60, peak_time=55, peak=1.0, mean=0.8, active=10)
    text = Event(start=94, end=101, peak_time=100, peak=3.0, boost=95, source="text",
                 reasons=('on screen: "penta kill"',))  # fmt: skip
    sel = select_events([audio, text], duration=300, merge_gap=4, pad_before=6, pad_after=4,
                        min_len=8, max_len=60, top=2)  # fmt: skip
    best = max(sel.kept, key=lambda e: e.score)
    assert best.score >= 95 and best.peak_time == 100
    assert any("penta kill" in r for r in best.reasons)


def test_config_parsing_of_text_settings(tmp_path: Path) -> None:
    league = load_settings("league")
    assert league.text_scan and league.text_regions[1][4:] == (0.5, 18.0)
    victory = next(k for k in league.text_events if k.phrase == "victory")
    assert victory.team and victory.cooldown == 300
    assert not load_settings("generic").text_scan
    cfg = tmp_path / "c.toml"
    cfg.write_text("[defaults]\ntext_regions = [[0.5, 0.0, 0.2, 1.0]]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="text_regions"):
        load_settings(config_file=cfg)
    cfg.write_text('[presets.generic.text_events]\n"x" = {cooldown = 3}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="text_events"):
        load_settings(config_file=cfg)


def test_other_players_kills_do_not_chain_into_one_moment() -> None:
    """Regression: in a busy ARAM a different double kill appears every few seconds;
    each is its own moment and the player's own kill keeps its real time."""
    kills = TextKeyword("double kill", 65)
    names = ["Kuka65 (Illaoi)", "SHiRo (Shyvana)", "Dirk (Ashe)", "Laurucu (Kai'Sa)",
             "MatiPelusas (Ryze)", "Mackepurple (Vayne)", "arcamatis (Twisted Fate)"]  # fmt: skip
    pairs = [(a, b) for a in names for b in names if a != b][:40]
    hits = [
        T.TextHit(600.0 + 10 * i, kills, f"{a} has slain {b} for a double kill!")
        for i, (a, b) in enumerate(pairs)
    ]
    hits.append(T.TextHit(1125.0, kills, REAL_LINES["mine"].replace("triple", "double"), mine=True))
    events = T.hits_to_events(hits, PLAYER)
    # Before the fix all of these chained into ~2 "moments" spanning ten minutes.
    assert len(events) >= 0.75 * len(hits)
    assert max(e.peak_time for e in events if "(you)" not in e.reasons[0]) > 900
    mine = [e for e in events if "(you)" in e.reasons[0]]
    assert [e.peak_time for e in mine] == [1125.0]


def test_line_identity_matches_rereads_not_other_kills() -> None:
    from difflib import SequenceMatcher

    def same(a: str, b: str) -> float:
        return SequenceMatcher(
            None, T.line_identity(a, "triple kill"), T.line_identity(b, "triple kill")
        ).ratio()

    assert same(REAL_LINES["mine"], REAL_LINES["mine_again"]) >= T.SAME_LINE_RATIO
    assert same(REAL_LINES["mine"], REAL_LINES["other"]) < T.SAME_LINE_RATIO
