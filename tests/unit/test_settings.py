"""Tests for the configuration module."""

from pathlib import Path

import pytest

from agentic_rag.config.settings import (
    CorpusSettings,
    LoggingSettings,
    PathSettings,
    Settings,
    get_settings,
)


def test_settings_defaults_are_valid() -> None:
    settings = Settings()
    assert settings.environment == "local"
    assert settings.random_seed == 42


def test_get_settings_returns_cached_instance() -> None:
    assert get_settings() is get_settings()


def test_log_level_is_normalised_to_uppercase() -> None:
    assert LoggingSettings(level="debug").level == "DEBUG"  # type: ignore[arg-type]


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        LoggingSettings(level="VERBOSE")  # type: ignore[arg-type]


def test_min_document_chars_rejects_negative() -> None:
    with pytest.raises(ValueError):
        CorpusSettings(min_document_chars=-1)


def test_ensure_exists_creates_directories(tmp_path: Path) -> None:
    paths = PathSettings(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        processed_dir=tmp_path / "data" / "processed",
        evalsets_dir=tmp_path / "data" / "evalsets",
    )
    paths.ensure_exists()
    assert paths.raw_dir.is_dir()
    assert paths.evalsets_dir.is_dir()
