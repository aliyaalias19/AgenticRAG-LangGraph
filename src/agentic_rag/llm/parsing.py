"""Helpers for parsing structured output from model responses."""

import json
import re
from typing import Any

FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class ParseError(ValueError):
    """Raised when a model response cannot be parsed as the expected JSON."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating markdown fences."""
    cleaned = FENCE_PATTERN.sub("", text).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as error:
        message = f"Response is not valid JSON: {error}"
        raise ParseError(message) from error

    if not isinstance(parsed, dict):
        message = f"Expected a JSON object, got {type(parsed).__name__}"
        raise ParseError(message)

    return parsed


def parse_string_list(text: str, key: str) -> list[str]:
    """Parse a JSON object and return its list-of-strings value at ``key``."""
    parsed = parse_json_object(text)

    if key not in parsed:
        message = f"Response is missing the key {key!r}"
        raise ParseError(message)

    value = parsed[key]
    if not isinstance(value, list):
        message = f"Expected a list at {key!r}, got {type(value).__name__}"
        raise ParseError(message)

    items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if not items:
        message = f"No non-empty strings found at {key!r}"
        raise ParseError(message)

    return items
