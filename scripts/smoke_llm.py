"""Verify the Anthropic client works against the real API."""

from agentic_rag.llm.client import LLMClient
from agentic_rag.obs.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

client = LLMClient()

response = client.complete(
    "Reply with exactly the word: pong",
    system="You follow instructions literally.",
    max_tokens=16,
)

print(f"response: {response!r}")
print(f"usage   : {client.usage.as_dict()}")

cached = client.complete(
    "Reply with exactly the word: pong",
    system="You follow instructions literally.",
    max_tokens=16,
)
print(f"cached  : {cached!r}")
print(f"usage   : {client.usage.as_dict()}")
