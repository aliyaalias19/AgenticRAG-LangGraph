"""Smoke tests verifying the package is installed and importable."""

from __future__ import annotations

import importlib.metadata

import agentic_rag


def test_package_is_importable() -> None:
    assert agentic_rag is not None


def test_package_has_version() -> None:
    version = importlib.metadata.version("agentic-rag")
    assert version == "0.1.0"
