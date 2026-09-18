"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PathSettings(BaseSettings):
    """Filesystem locations for corpora, artifacts, and evaluation sets."""

    model_config = SettingsConfigDict(env_prefix="PATH_", extra="ignore")

    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    evalsets_dir: Path = Field(default=PROJECT_ROOT / "data" / "evalsets")

    def ensure_exists(self) -> None:
        """Create all configured directories if they do not exist."""
        for directory in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.evalsets_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class CorpusSettings(BaseSettings):
    """Source and filtering rules for the document corpus."""

    model_config = SettingsConfigDict(env_prefix="CORPUS_", extra="ignore")

    repo_url: str = Field(default="https://github.com/kubernetes/website.git")
    repo_ref: str = Field(default="main")
    docs_subpath: str = Field(default="content/en/docs")
    min_document_chars: int = Field(default=200, ge=0)
    excluded_path_prefixes: tuple[str, ...] = Field(
        default=(
            "reference/kubernetes-api",
            "reference/instrumentation/metrics",
            "reference/command-line-tools-reference/feature-gates",
        ),
        description="Path prefixes excluded as auto-generated or stub content",
    )


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    json_output: bool = Field(default=False)

    @field_validator("level", mode="before")
    @classmethod
    def uppercase_level(cls, value: str) -> str:
        return value.upper() if isinstance(value, str) else value


class Settings(BaseSettings):
    """Root configuration object aggregating all settings groups."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["local", "ci", "production"] = Field(default="local")
    random_seed: int = Field(default=42)

    paths: PathSettings = Field(default_factory=PathSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
