"""Tests for the Anthropic client wrapper."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.config.settings import Settings
from agentic_rag.llm.client import LLMClient, UsageStats


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LLMClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with patch("agentic_rag.llm.client.anthropic.Anthropic"):
        return LLMClient(settings=Settings(), cache_dir=tmp_path / "cache")


def _fake_message(text: str, input_tokens: int = 10, output_tokens: int = 5) -> Any:
    block = MagicMock()
    block.type = "text"
    block.text = text

    message = MagicMock()
    message.content = [block]
    message.usage.input_tokens = input_tokens
    message.usage.output_tokens = output_tokens
    return message


class TestUsageStats:
    def test_record_accumulates_tokens(self) -> None:
        usage = UsageStats()
        usage.record(input_tokens=100, output_tokens=50)
        usage.record(input_tokens=200, output_tokens=25)

        assert usage.calls == 2
        assert usage.input_tokens == 300
        assert usage.output_tokens == 75


class TestClientConstruction:
    def test_missing_api_key_raises(self, tmp_path: Path) -> None:
        settings = Settings()
        settings.llm.api_key = ""

        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            LLMClient(settings=settings, cache_dir=tmp_path)

    def test_cache_directory_is_created(self, client: LLMClient) -> None:
        assert client.cache_dir is not None
        assert client.cache_dir.is_dir()


class TestCaching:
    def test_identical_prompts_share_a_cache_key(self, client: LLMClient) -> None:
        first = client._cache_key("sys", "prompt")
        second = client._cache_key("sys", "prompt")
        assert first == second

    def test_different_prompts_differ(self, client: LLMClient) -> None:
        assert client._cache_key("sys", "a") != client._cache_key("sys", "b")

    def test_different_systems_differ(self, client: LLMClient) -> None:
        assert client._cache_key("a", "prompt") != client._cache_key("b", "prompt")

    def test_second_call_is_served_from_cache(self, client: LLMClient) -> None:
        client._client.messages.create.return_value = _fake_message("answer")

        first = client.complete("question")
        second = client.complete("question")

        assert first == second == "answer"
        assert client._client.messages.create.call_count == 1
        assert client.usage.cache_hits == 1

    def test_cache_survives_a_new_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        cache_dir = tmp_path / "cache"

        with patch("agentic_rag.llm.client.anthropic.Anthropic"):
            first = LLMClient(settings=Settings(), cache_dir=cache_dir)
            first._client.messages.create.return_value = _fake_message("answer")
            first.complete("question")
            calls_after_first = first._client.messages.create.call_count

            second = LLMClient(settings=Settings(), cache_dir=cache_dir)
            assert second.complete("question") == "answer"
            assert second._client.messages.create.call_count == calls_after_first
            assert second.usage.cache_hits == 1

    def test_caching_can_be_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        settings = Settings()
        settings.llm.cache_enabled = False

        with patch("agentic_rag.llm.client.anthropic.Anthropic"):
            client = LLMClient(settings=settings, cache_dir=tmp_path / "cache")
            client._client.messages.create.return_value = _fake_message("answer")

            client.complete("question")
            client.complete("question")

            assert client._client.messages.create.call_count == 2


class TestCompletion:
    def test_usage_is_recorded(self, client: LLMClient) -> None:
        client._client.messages.create.return_value = _fake_message(
            "answer", input_tokens=120, output_tokens=40
        )
        client.complete("question")

        assert client.usage.calls == 1
        assert client.usage.input_tokens == 120
        assert client.usage.output_tokens == 40

    def test_system_prompt_is_omitted_when_empty(self, client: LLMClient) -> None:
        client._client.messages.create.return_value = _fake_message("answer")
        client.complete("question")

        kwargs = client._client.messages.create.call_args.kwargs
        assert "system" not in kwargs

    def test_system_prompt_is_passed_when_provided(self, client: LLMClient) -> None:
        client._client.messages.create.return_value = _fake_message("answer")
        client.complete("question", system="You are terse.")

        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["system"] == "You are terse."

    def test_multiple_text_blocks_are_joined(self, client: LLMClient) -> None:
        first, second = MagicMock(), MagicMock()
        first.type, first.text = "text", "part one"
        second.type, second.text = "text", "part two"

        message = MagicMock()
        message.content = [first, second]
        message.usage.input_tokens = 10
        message.usage.output_tokens = 5
        client._client.messages.create.return_value = message

        assert client.complete("question") == "part one\npart two"

    def test_cache_file_contains_the_response(self, client: LLMClient) -> None:
        client._client.messages.create.return_value = _fake_message("answer")
        client.complete("question")

        assert client.cache_dir is not None
        files = list(client.cache_dir.glob("*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["response"] == "answer"
