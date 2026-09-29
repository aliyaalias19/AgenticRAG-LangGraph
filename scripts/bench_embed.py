"""Measure embedding throughput at different max_length settings."""

import time

import torch

from agentic_rag.config.settings import get_settings
from agentic_rag.retrieval.embedder import Embedder

print(f"cpu cores    : {torch.get_num_threads()}")
torch.set_num_threads(torch.get_num_threads())

settings = get_settings()
embedder = Embedder(settings=settings)

# Warm up so model init is excluded from the measurement.
embedder.encode(["warmup"])

sample = ["A Kubernetes pod is the smallest deployable unit. " * 20] * 16

for max_length in (256, 512, 1024):
    settings.embedding.max_length = max_length
    start = time.perf_counter()
    embedder.encode(sample)
    elapsed = time.perf_counter() - start
    per_chunk = elapsed / len(sample)
    print(
        f"max_length={max_length:>5}  "
        f"{elapsed:6.1f}s for 16  "
        f"{per_chunk:5.2f}s/chunk  "
        f"est. 15081 chunks: {per_chunk * 15081 / 3600:.1f}h"
    )
