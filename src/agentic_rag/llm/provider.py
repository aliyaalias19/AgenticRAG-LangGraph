"""Inference provider abstraction.

Orchestration code depends only on ``ChatProvider``. Swapping a local Ollama
model for a hosted API, or for a self-hosted vLLM endpoint, changes one
settings value and touches no graph logic. Every provider returns the same
``Completion`` object, so token accounting and cost reporting work identically
whichever backend is active.
"""

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from agentic_rag.config.settings import Settings, get_settings
from agentic_rag.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class Completion:
    """A model response with its token accounting."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ProviderUsage:
    """Cumulative usage for one provider instance."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, completion: Completion) -> None:
        self.calls += 1
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class ChatProvider(ABC):
    """Interface every inference backend implements."""

    name: str = "abstract"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.usage = ProviderUsage()

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Completion:
        """Return a completion for the given messages."""

    def stream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield response text incrementally.

        The default implementation falls back to a single non-streaming call,
        so a provider without streaming support still satisfies the interface.
        """
        yield self.complete(messages, system=system, max_tokens=max_tokens).text

    def _record(self, completion: Completion) -> Completion:
        self.usage.record(completion)
        return completion


class AnthropicProvider(ChatProvider):
    """Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        import anthropic

        api_key = self.settings.llm.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            message = "ANTHROPIC_API_KEY is not set"
            raise ValueError(message)
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.settings.llm.model,
            "max_tokens": max_tokens or self.settings.llm.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        text = "\n".join(b.text for b in response.content if b.type == "text")
        return self._record(
            Completion(
                text=text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=self.settings.llm.model,
                provider=self.name,
            )
        )

    def stream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "model": self.settings.llm.model,
            "max_tokens": max_tokens or self.settings.llm.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream


class OllamaProvider(ChatProvider):
    """Local inference through an Ollama server."""

    name = "ollama"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        import httpx

        self._http = httpx.Client(base_url=self.settings.llm.ollama_base_url, timeout=300.0)

    def _payload(
        self, messages: list[Message], system: str, max_tokens: int | None
    ) -> dict[str, Any]:
        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend({"role": m.role, "content": m.content} for m in messages)
        return {
            "model": self.settings.llm.ollama_model,
            "messages": chat,
            "options": {"num_predict": max_tokens or self.settings.llm.max_tokens},
        }

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Completion:
        payload = self._payload(messages, system, max_tokens)
        payload["stream"] = False
        response = self._http.post("/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()
        return self._record(
            Completion(
                text=body["message"]["content"],
                input_tokens=int(body.get("prompt_eval_count", 0)),
                output_tokens=int(body.get("eval_count", 0)),
                model=self.settings.llm.ollama_model,
                provider=self.name,
            )
        )

    def stream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = self._payload(messages, system, max_tokens)
        payload["stream"] = True
        with self._http.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                body = json.loads(line)
                chunk = body.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if body.get("done"):
                    break


class OpenAICompatibleProvider(ChatProvider):
    """Any OpenAI-compatible chat completions endpoint.

    This covers the hosted OpenAI API and a self-hosted vLLM server, which
    exposes the same schema. Only the base URL and model name differ, so the
    fine-tuned model served through vLLM is reachable with no code change.
    """

    name = "openai"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(settings)
        import httpx

        self._model = model or self.settings.llm.openai_model
        key = api_key if api_key is not None else self.settings.llm.openai_api_key
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.Client(
            base_url=base_url or self.settings.llm.openai_base_url,
            headers=headers,
            timeout=300.0,
        )

    def _payload(
        self, messages: list[Message], system: str, max_tokens: int | None
    ) -> dict[str, Any]:
        chat: list[dict[str, str]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend({"role": m.role, "content": m.content} for m in messages)
        return {
            "model": self._model,
            "messages": chat,
            "max_tokens": max_tokens or self.settings.llm.max_tokens,
        }

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Completion:
        payload = self._payload(messages, system, max_tokens)
        response = self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        return self._record(
            Completion(
                text=body["choices"][0]["message"]["content"],
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                model=self._model,
                provider=self.name,
            )
        )

    def stream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = self._payload(messages, system, max_tokens)
        payload["stream"] = True
        with self._http.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if content := delta.get("content"):
                    yield content


class BedrockProvider(ChatProvider):
    """AWS Bedrock Converse API."""

    name = "bedrock"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=self.settings.llm.bedrock_region)

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "modelId": self.settings.llm.bedrock_model,
            "messages": [{"role": m.role, "content": [{"text": m.content}]} for m in messages],
            "inferenceConfig": {"maxTokens": max_tokens or self.settings.llm.max_tokens},
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self._client.converse(**kwargs)
        text = "".join(block.get("text", "") for block in response["output"]["message"]["content"])
        usage = response.get("usage", {})
        return self._record(
            Completion(
                text=text,
                input_tokens=int(usage.get("inputTokens", 0)),
                output_tokens=int(usage.get("outputTokens", 0)),
                model=self.settings.llm.bedrock_model,
                provider=self.name,
            )
        )


@dataclass
class EchoProvider(ChatProvider):
    """Deterministic provider used in tests and offline development."""

    name: str = "echo"
    responses: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    systems: list[str] = field(default_factory=list)

    def __init__(
        self, responses: list[str] | None = None, settings: Settings | None = None
    ) -> None:
        super().__init__(settings)
        self.name = "echo"
        self.responses = list(responses or [])
        self.prompts = []
        self.systems = []
        self._position = 0

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> Completion:
        del max_tokens
        self.prompts.append(messages[-1].content if messages else "")
        self.systems.append(system)

        if self._position < len(self.responses):
            text = self.responses[self._position]
            self._position += 1
        else:
            text = self.responses[-1] if self.responses else ""

        return self._record(
            Completion(
                text=text,
                input_tokens=sum(len(m.content) // 4 for m in messages),
                output_tokens=len(text) // 4,
                model="echo",
                provider=self.name,
            )
        )


PROVIDERS: dict[str, type[ChatProvider]] = {
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "openai": OpenAICompatibleProvider,
    "bedrock": BedrockProvider,
}


def build_provider(settings: Settings | None = None) -> ChatProvider:
    """Construct the provider named in settings."""
    settings = settings or get_settings()
    choice = settings.llm.provider

    if choice == "vllm":
        return OpenAICompatibleProvider(
            settings,
            base_url=settings.llm.vllm_base_url,
            model=settings.llm.vllm_model,
            api_key="",
        )

    provider_class = PROVIDERS.get(choice)
    if provider_class is None:
        message = f"Unknown provider: {choice}"
        raise ValueError(message)

    logger.info("provider_selected", provider=choice)
    return provider_class(settings)
