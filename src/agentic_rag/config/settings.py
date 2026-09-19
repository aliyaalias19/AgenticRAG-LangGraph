"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
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


class SourceSettings(BaseSettings):
    """A single documentation source repository."""

    model_config = SettingsConfigDict(extra="ignore")

    name: str = Field(description="Short identifier, e.g. 'kubernetes'")
    repo_url: str
    repo_ref: str = Field(default="main")
    docs_subpath: str
    language: Literal["en", "zh"] = Field(default="en")
    excluded_path_prefixes: tuple[str, ...] = Field(default=())


class CorpusSettings(BaseSettings):
    """Corpus-wide ingestion rules."""

    model_config = SettingsConfigDict(env_prefix="CORPUS_", extra="ignore")

    min_document_chars: int = Field(default=200, ge=0)
    sources: tuple[SourceSettings, ...] = Field(
        default=(
            SourceSettings(
                name="kubernetes",
                repo_url="https://github.com/kubernetes/website.git",
                docs_subpath="content/en/docs",
                language="en",
                excluded_path_prefixes=(
                    "reference/kubernetes-api",
                    "reference/instrumentation/metrics",
                    "reference/command-line-tools-reference/feature-gates",
                    "test",
                ),
            ),
            SourceSettings(
                name="kubernetes-zh",
                repo_url="https://github.com/kubernetes/website.git",
                docs_subpath="content/zh-cn/docs",
                language="zh",
                excluded_path_prefixes=(
                    "reference/kubernetes-api",
                    "reference/instrumentation/metrics",
                    "reference/command-line-tools-reference/feature-gates",
                    "test",
                ),
            ),
        )
    )


class ChunkSettings(BaseSettings):
    """Document chunking parameters."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")

    max_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=150, ge=0)
    min_chars: int = Field(default=100, ge=0)
    max_heading_depth: int = Field(default=3, ge=1, le=6)

    @field_validator("overlap_chars")
    @classmethod
    def overlap_must_be_smaller_than_max(cls, value: int, info: ValidationInfo) -> int:
        max_chars = info.data.get("max_chars")
        if max_chars is not None and value >= max_chars:
            message = "overlap_chars must be smaller than max_chars"
            raise ValueError(message)
        return value


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
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
