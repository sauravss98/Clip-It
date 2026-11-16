from __future__ import annotations

from pathlib import Path

import pytest

from game_clipper.config import ConfigError, available_presets, load_settings

EXPECTED = {"generic", "fortnite", "league", "r6siege", "gta5_online", "gta6"}


def test_all_presets_load() -> None:
    assert set(available_presets()) >= EXPECTED
    for name in EXPECTED:
        s = load_settings(name)
        assert s.w_sustained + s.w_transient == pytest.approx(1.0)


def test_precedence(tmp_path: Path) -> None:
    cfg = tmp_path / "my config.toml"
    cfg.write_text(
        "[defaults]\nthreshold = 0.9\npad_after = 2\n"
        "[presets.r6siege]\nmerge_gap = 1.5\n"
        "[presets.mygame]\ndescription = 'custom'\nw_sustained = 0.5\nw_transient = 0.5\n",
        encoding="utf-8",
    )
    s = load_settings("r6siege", cfg, {"top": 3, "pad_before": None})
    assert s.pad_after == 2  # user defaults beat built-in defaults
    assert s.threshold == 0.55  # built-in preset beats user defaults
    assert s.merge_gap == 1.5  # user preset beats built-in preset
    assert s.top == 3 and s.pad_before == 6.0  # CLI wins; None is ignored
    assert load_settings("mygame", cfg).w_sustained == 0.5
    assert "mygame" in available_presets(cfg)


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown preset"):
        load_settings("minecraft")
    with pytest.raises(ConfigError, match="Unknown setting"):
        load_settings(overrides={"nope": 1})
    with pytest.raises(ConfigError, match="min_len"):
        load_settings(overrides={"min_len": 90, "max_len": 60})
    bad = tmp_path / "bad.toml"
    bad.write_text("[defaults\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="parse"):
        load_settings(config_file=bad)
