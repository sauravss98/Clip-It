"""On-screen text detector: reads kill feeds, chat and banners with local OCR.

Audio can't always tell a triple kill from walking around: stream VODs often have
the game mixed under music, and MOBAs are noisy all the time. But games *print*
their big moments: "has slain X for a triple kill!", "VICTORY", "ELIMINATED".
This detector samples a frame every few seconds, reads only the regions where such
text appears (configured per preset) and turns keyword hits into events.

OCR runs locally on the CPU with RapidOCR (ONNX); install the ``[ocr]`` extra.
To stay fast enough for long recordings only small screen regions are read, banner
regions are shrunk and skip small text, and frames are sampled every few seconds
(kill lines stay in chat for ~10 s, so nothing is missed).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from game_clipper.config import Settings, TextKeyword
from game_clipper.detectors.base import Event
from game_clipper.ffmpeg import FFmpegError, probe

INSTALL_HINT = 'Reading on-screen text needs the optional extra: pip install -e ".[ocr]"'
FRAME_HEIGHT = 720  # frames are scaled to this height before reading
BEFORE, AFTER = 6.0, 1.0  # event span around the moment the text appears (before padding)
MINE_BONUS, OTHERS_FACTOR = 10.0, 0.8
SAME_LINE_WINDOW = 180.0  # a near-identical line this soon is the same moment
SAME_LINE_RATIO = 0.8  # similarity of two OCR reads of the same line (names only)

# Characters OCR routinely confuses, folded together before matching.
_CONFUSABLE = str.maketrans({"0": "o", "1": "l", "i": "l", "|": "l", "!": "l", "5": "s"})


def available() -> bool:
    """True when the OCR engine is installed."""
    return find_spec("rapidocr_onnxruntime") is not None


def normalize(text: str) -> str:
    """Fold case, OCR look-alikes, punctuation, spaces and repeated letters.

    ``"has slain X for a triple killl!"`` and ``"fora triplekil"`` both contain
    ``"trlplekl"`` afterwards, so simple substring search survives typical OCR noise.
    """
    folded = re.sub(r"[^a-z0-9|!]", "", text.lower()).translate(_CONFUSABLE)
    folded = re.sub(r"[^a-z0-9]", "", folded)
    return re.sub(r"(.)\1+", r"\1", folded)


def find_phrase(text_n: str, phrase_n: str, min_ratio: float = 0.85) -> int:
    """Position of ``phrase_n`` in ``text_n`` (both normalised), or -1.

    Exact substring first; long phrases also allow one or two OCR mistakes.
    """
    pos = text_n.find(phrase_n)
    if pos >= 0 or len(phrase_n) < 7:
        return pos
    # Cheap prefilter: a fuzzy match must share the phrase's start or end.
    if phrase_n[:3] not in text_n and phrase_n[-3:] not in text_n:
        return -1
    n = len(phrase_n)
    for size in (n, n - 1, n + 1):
        for i in range(0, max(1, len(text_n) - size + 1)):
            if SequenceMatcher(None, phrase_n, text_n[i : i + size]).ratio() >= min_ratio:
                return i
    return -1


@dataclass(frozen=True)
class TextHit:
    """A keyword read on screen at a moment of the video."""

    time: float
    keyword: TextKeyword
    text: str
    mine: bool = False


def match_line(
    text: str, time: float, keywords: Sequence[TextKeyword], player: str = ""
) -> list[TextHit]:
    """Keyword hits in one line of on-screen text.

    A line counts as the player's own when it *starts* with their name: kill feeds
    and chat read "KILLER has slain VICTIM ...", so a name later in the line is
    usually the victim.
    """
    text_n = normalize(text)
    player_n = normalize(player) if player else ""
    hits = []
    for kw in keywords:
        if find_phrase(text_n, normalize(kw.phrase)) >= 0:
            mine = bool(player_n) and 0 <= find_phrase(text_n[: len(player_n) + 4], player_n) <= 3
            hits.append(TextHit(time, kw, text.strip(), mine))
    return hits


# Words every kill line shares; removed so two lines are compared by who killed whom.
_FILLER = tuple(normalize(w) for w in ("has slain", "has shut down", "for a", "bonus bounty"))


def line_identity(text: str, phrase: str) -> str:
    """What makes a line unique: the text minus the keyword and shared filler words.

    ``"A (Ashe) has slain B (Senna) for a double kill!"`` -> roughly ``"aashebsenna"``,
    so two different double kills differ, while re-reads of one line still match.
    """
    text_n = normalize(text)
    phrase_n = normalize(phrase)
    pos = find_phrase(text_n, phrase_n)
    if pos >= 0:
        text_n = text_n[:pos] + text_n[pos + len(phrase_n) :]
    for filler in _FILLER:
        text_n = text_n.replace(filler, "")
    return text_n


def hits_to_events(hits: Sequence[TextHit], player: str = "") -> list[Event]:
    """Group repeated sightings of the same keyword into one event per moment.

    A chat line stays visible for ~10 s (or minutes, in chat history) and is read in
    several samples with slightly different typos each time. Near-identical lines
    seen within :data:`SAME_LINE_WINDOW` seconds (or the keyword's cooldown, if
    longer) are one moment, dated at the first sighting.
    """
    groups: list[list[TextHit]] = []
    for hit in sorted(hits, key=lambda h: h.time):
        line_n = line_identity(hit.text, hit.keyword.phrase)
        for group in reversed(groups):
            if group[0].keyword.phrase != hit.keyword.phrase:
                continue
            # Only the *same line* (re-read with different typos) is the same moment:
            # in a busy fight other players' kills with the same keyword appear every
            # few seconds and must not chain into one long "moment".
            window = max(hit.keyword.cooldown, SAME_LINE_WINDOW)
            if hit.time - group[-1].time <= window and any(
                SequenceMatcher(None, line_n, line_identity(h.text, h.keyword.phrase)).ratio()
                >= SAME_LINE_RATIO
                for h in group
            ):
                group.append(hit)
                break
        else:
            groups.append([hit])

    out = []
    for group in groups:
        first = group[0]
        mine = any(h.mine for h in group)
        score = first.keyword.score
        if player and not first.keyword.team:
            score = min(100.0, score + MINE_BONUS) if mine else score * OTHERS_FACTOR
        quote = (next(h for h in group if h.mine) if mine else first).text
        reason = f'on screen{" (you)" if mine else ""}: "{quote[:90]}"'
        out.append(
            Event(
                start=max(0.0, first.time - BEFORE),
                end=first.time + AFTER,
                peak_time=first.time,
                peak=3.0,  # wins peak_time when merged with audio events
                boost=round(score, 1),
                reasons=(reason,),
                source="text",
            )
        )
    return out


class LineReader(Protocol):
    """Reads the text lines in an image (RapidOCR in production, fakes in tests)."""

    def __call__(self, image: np.ndarray, min_height: float = 0.0) -> list[str]: ...


class RapidReader:
    """Line reader backed by RapidOCR, loaded on first use."""

    def __init__(self) -> None:
        self._ocr: Any = None

    def __call__(self, image: np.ndarray, min_height: float = 0.0) -> list[str]:
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr = RapidOCR()
        ocr = self._ocr
        boxes, _ = ocr.text_detector(image)
        if boxes is None or len(boxes) == 0:
            return []
        boxes = ocr.sorted_boxes(boxes)
        if min_height:
            # Box corners: top-left, top-right, bottom-right, bottom-left.
            boxes = [b for b in boxes if np.linalg.norm(b[3] - b[0]) >= min_height]
            if not boxes:
                return []
        crops = ocr.get_crop_img_list(image, boxes)
        results, _ = ocr.text_recognizer(crops)
        return [text for text, score in results if float(score) >= 0.5 and text.strip()]


def sample_frames(
    video: Path, interval: float, duration: float, width: int, height: int
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(time, BGR frame)`` every ``interval`` seconds, scaled to 720p height.

    The end of the video is always sampled too: VICTORY banners live in the last
    seconds of a recording.
    """
    h = FRAME_HEIGHT
    w = max(2, round(width * h / max(height, 1) / 2) * 2)
    size = w * h * 3
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-hwaccel", "auto",
        "-i", str(video), "-map", "0:v:0", "-vf", f"fps=1/{interval},scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    last = -interval
    try:
        i = 0
        while len(buf := proc.stdout.read(size)) == size:
            last = i * interval
            yield last, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()
    for t in (duration - 2.0, duration - 0.5):
        if t > last + 0.5:
            one = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(video),
                 "-map", "0:v:0", "-frames:v", "1", "-vf", f"scale={w}:{h}",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                capture_output=True,
            )  # fmt: skip
            if len(one.stdout) == size:
                last = t
                yield t, np.frombuffer(one.stdout, np.uint8).reshape(h, w, 3)


def crop_regions(
    frame: np.ndarray, regions: Sequence[tuple[float, ...]]
) -> list[tuple[np.ndarray, float]]:
    """Cut the configured regions out of ``frame``: ``[(crop, min_text_px), ...]``.

    Regions are ``(x0, y0, x1, y1[, scale, min_text_px])`` in fractions of the frame.
    """
    h, w = frame.shape[:2]
    if not regions:
        return [(frame, 0.0)]
    out = []
    for region in regions:
        x0, y0, x1, y1 = region[:4]
        scale = region[4] if len(region) > 4 else 1.0
        min_px = region[5] if len(region) > 5 else 0.0
        crop = frame[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
        if scale < 1.0:
            step = max(1, round(1 / scale))
            crop = crop[::step, ::step]  # cheap downscale; banner text is large
        out.append((np.ascontiguousarray(crop), min_px * scale))
    return out


class TextDetector:
    """Detector that finds big moments by reading on-screen text."""

    name = "text"

    def __init__(
        self,
        reader_factory: Callable[[], LineReader] = RapidReader,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        self.reader_factory = reader_factory
        self.on_progress = on_progress
        self.hits: list[TextHit] = []

    def scan(
        self,
        frames: Iterator[tuple[float, np.ndarray]],
        settings: Settings,
    ) -> list[Event]:
        """Read every sampled frame and return events for keyword hits."""
        reader = self.reader_factory()
        self.hits = []
        for t, frame in frames:
            for crop, min_px in crop_regions(frame, settings.text_regions):
                for line in reader(crop, min_px):
                    self.hits += match_line(line, t, settings.text_events, settings.player_name)
            if self.on_progress:
                self.on_progress(t)
        return hits_to_events(self.hits, settings.player_name)

    def detect(self, video: Path, settings: Settings) -> list[Event]:
        """Scan ``video`` for the preset's on-screen keywords."""
        if not settings.text_events:
            return []
        info = probe(video)
        if not info.has_video or not info.width or not info.height:
            raise FFmpegError(f"{video.name} has no video stream to read text from.")
        frames = sample_frames(
            video, settings.text_interval, info.duration, info.width, info.height
        )
        return self.scan(frames, settings)
