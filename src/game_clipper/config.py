"""Preset and config loading.

Presets only change parameters and weights; there are no game-specific code paths.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, fields
from importlib import resources
from pathlib import Path
from typing import Any

DEFAULT_PRESET = "generic"


class ConfigError(ValueError):
    """Raised for unknown presets, unknown keys or invalid values."""


@dataclass(frozen=True)
class TextKeyword:
    """An on-screen phrase worth clipping, e.g. "triple kill" or "victory royale"."""

    phrase: str
    score: float  # 0-100 clip score when it appears
    cooldown: float = 15.0  # repeats within this many seconds belong to the same moment
    team: bool = False  # team-wide moment (victory, ace): never discounted as "not yours"


@dataclass(frozen=True)
class Settings:
    """All tunable parameters for detection and event selection."""

    description: str = ""
    sample_rate: int = 16000
    window: float = 0.5
    baseline_seconds: float = 90.0
    baseline_percentile: float = 40.0
    w_sustained: float = 0.6
    w_transient: float = 0.4
    transient_lookback: float = 3.0
    threshold: float = 0.6
    max_flagged: float = 0.08
    sustain_window: float = 2.0
    music_floor_db: float = 6.0
    music_min_len: float = 20.0
    music_max_std_db: float = 4.0
    music_penalty: float = 0.3
    tonal_penalty: float = 0.0
    merge_gap: float = 4.0
    pad_before: float = 6.0
    pad_after: float = 4.0
    min_len: float = 8.0
    max_len: float = 60.0
    top: int = 20
    # On-screen text (kill feed, chat, banners). Needs the [ocr] extra.
    text_scan: bool = False
    text_interval: float = 3.0
    text_regions: tuple[tuple[float, ...], ...] = ()  # (x0, y0, x1, y1, scale, min_text_px)
    text_events: tuple[TextKeyword, ...] = ()
    player_name: str = ""

    def __post_init__(self) -> None:
        if self.window <= 0 or self.sample_rate <= 0:
            raise ConfigError("window and sample_rate must be positive")
        if self.min_len > self.max_len:
            raise ConfigError(f"min_len ({self.min_len}) is longer than max_len ({self.max_len})")
        if self.top < 1:
            raise ConfigError("top must be at least 1")
        if self.max_flagged <= 0:
            raise ConfigError("max_flagged must be positive (1 disables the cap)")
        for name in ("music_penalty", "tonal_penalty"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ConfigError(f"{name} must be between 0 and 1")
        if not 0.0 <= self.baseline_percentile <= 100.0:
            raise ConfigError("baseline_percentile must be between 0 and 100")
        if self.text_interval <= 0:
            raise ConfigError("text_interval must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a plain dict (for events.json)."""
        return asdict(self)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in (
        "true",
        "1",
        "yes",
        "on",
        "false",
        "0",
        "no",
        "off",
    ):
        return value.lower() in ("true", "1", "yes", "on")
    raise ValueError(value)


def _to_regions(value: Any) -> tuple[tuple[float, ...], ...]:
    """``[[x0, y0, x1, y1], [x0, y0, x1, y1, scale, min_text_px], ...]``.

    Coordinates are fractions of the frame. ``scale`` shrinks the region before
    reading (fine for big banner text) and ``min_text_px`` skips text lines shorter
    than that many pixels (names and damage numbers over characters).
    """
    regions = []
    for box in value:
        nums = [float(v) for v in box]
        if len(nums) == 4:
            nums += [1.0, 0.0]
        if len(nums) != 6:
            raise ValueError(box)
        x0, y0, x1, y1, scale, min_px = nums
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and 0 < scale <= 1 and min_px >= 0):
            raise ValueError(box)
        regions.append((x0, y0, x1, y1, scale, min_px))
    return tuple(regions)


def _to_keywords(value: Any) -> tuple[TextKeyword, ...]:
    """``{"triple kill" = 90, "victory" = {score = 85, cooldown = 180, team = true}}``."""
    if not isinstance(value, dict):
        raise ValueError(value)
    out = []
    for phrase, spec in value.items():
        if isinstance(spec, dict):
            out.append(
                TextKeyword(
                    str(phrase),
                    float(spec["score"]),
                    float(spec.get("cooldown", 15)),
                    _to_bool(spec.get("team", False)),
                )
            )
        else:
            out.append(TextKeyword(str(phrase), float(spec)))
    return tuple(out)


_FIELD_TYPES = {f.name: str(f.type) for f in fields(Settings)}
_CASTS: dict[str, Any] = {"int": int, "float": float, "str": str, "bool": _to_bool}
_FIELD_CASTS: dict[str, Any] = {"text_regions": _to_regions, "text_events": _to_keywords}


def _coerce(values: dict[str, Any], source: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key not in _FIELD_TYPES:
            raise ConfigError(f"Unknown setting '{key}' in {source}")
        cast = _FIELD_CASTS.get(key) or _CASTS[_FIELD_TYPES[key]]
        try:
            out[key] = cast(value)
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigError(f"Invalid value for '{key}' in {source}: {value!r}") from exc
    return out


def _read_toml(text: str, source: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Could not parse {source}: {exc}") from exc


def builtin_config() -> dict[str, Any]:
    """Load the presets.toml shipped with the package."""
    text = resources.files("game_clipper").joinpath("presets.toml").read_text(encoding="utf-8")
    return _read_toml(text, "built-in presets.toml")


def available_presets(config_file: Path | None = None) -> dict[str, str]:
    """Map preset name to description, including presets added by ``config_file``."""
    layers = [builtin_config()]
    if config_file is not None:
        layers.append(_read_toml(config_file.read_text(encoding="utf-8"), str(config_file)))
    names: dict[str, str] = {}
    for layer in layers:
        for name, values in layer.get("presets", {}).items():
            names[name] = values.get("description", names.get(name, ""))
    return names


def load_settings(
    preset: str = DEFAULT_PRESET,
    config_file: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Resolve settings from built-in defaults, presets, a user config and CLI overrides.

    Order (later wins): built-in defaults, user defaults, built-in preset,
    user preset, then ``overrides`` (``None`` values are ignored).
    """
    builtin = builtin_config()
    user: dict[str, Any] = {}
    if config_file is not None:
        if not config_file.is_file():
            raise ConfigError(f"Config file not found: {config_file}")
        user = _read_toml(config_file.read_text(encoding="utf-8"), str(config_file))

    builtin_presets = builtin.get("presets", {})
    user_presets = user.get("presets", {})
    if preset not in builtin_presets and preset not in user_presets:
        known = ", ".join(sorted({*builtin_presets, *user_presets}))
        raise ConfigError(f"Unknown preset '{preset}'. Available: {known}")

    merged: dict[str, Any] = {}
    merged |= _coerce(builtin.get("defaults", {}), "built-in defaults")
    merged |= _coerce(user.get("defaults", {}), f"{config_file} [defaults]")
    merged |= _coerce(builtin_presets.get(preset, {}), f"built-in preset '{preset}'")
    merged |= _coerce(user_presets.get(preset, {}), f"{config_file} [presets.{preset}]")
    merged |= _coerce({k: v for k, v in (overrides or {}).items() if v is not None}, "CLI")
    return Settings(**merged)
