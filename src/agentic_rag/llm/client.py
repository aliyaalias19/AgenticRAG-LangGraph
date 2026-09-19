"""Anthropic API client with retries, caching, and cost tracking."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)

RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


@dataclass
class UsageStats:
    """Cumulative token usage across a client's lifetime."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass
class LLMClient:
    """Thin wrapper over the Anthropic SDK with disk caching and retries."""

    settings: Settings = field(default_factory=get_settings)
    cache_dir: Path | None = None
    usage: UsageStats = field(default_factory=UsageStats)

    def __post_init__(self) -> None:
        api_key = self.settings.llm.api_key
        if not api_key:
            message = "ANTHROPIC_API_KEY is not set"
            raise ValueError(message)

        self._client = anthropic.Anthropic(api_key=api_key)
        if self.cache_dir is None:
            self.cache_dir = self.settings.paths.data_dir / "llm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, system: str, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.settings.llm.model,
                "system": system,
                "prompt": prompt,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> str | None:
        if not self.settings.llm.cache_enabled or self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        text: str = data["response"]
        return text

    def _write_cache(self, key: str, response: str) -> None:
        if not self.settings.llm.cache_enabled or self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.json"
        path.write_text(
            json.dumps({"response": response}, ensure_ascii=False),
            encoding="utf-8",
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """Return the model's text response, using the disk cache when possible."""
        key = self._cache_key(system, prompt)

        cached = self._read_cache(key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached

        response = self._call_api(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens or self.settings.llm.max_tokens,
        )
        self._write_cache(key, response)
        return response

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _call_api(self, prompt: str, system: str, max_tokens: int) -> str:
        kwargs: dict[str, Any] = {
            "model": self.settings.llm.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        message = self._client.messages.create(**kwargs)

        self.usage.record(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )

        parts = [block.text for block in message.content if block.type == "text"]
        return "\n".join(parts)
