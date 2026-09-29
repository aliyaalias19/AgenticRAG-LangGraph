"""Verify BGE-M3 loads and produces sensible embeddings."""

import time

from agentic_rag.obs.logging import configure_logging
from agentic_rag.retrieval.embedder import Embedder

configure_logging()

texts = [
    "How do I drain a node before maintenance?",
    "kubectl drain safely evicts all pods from a node",
    "The weather in Kuala Lumpur is warm and humid",
    "如何在维护前排空节点？",
]

embedder = Embedder()

start = time.perf_counter()
result = embedder.encode(texts)
elapsed = time.perf_counter() - start

print(f"encoded {len(result)} texts in {elapsed:.1f}s")
print(f"dense dim      : {len(result.dense[0])}")
print(f"sparse tokens  : {[len(s) for s in result.sparse]}")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


print("\ncosine similarity to query 0:")
for i, text in enumerate(texts[1:], start=1):
    score = cosine(result.dense[0], result.dense[i])
    print(f"  {score:.3f}  {text[:50]}")
