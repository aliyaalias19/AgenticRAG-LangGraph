"""Inspect chunks below the configured minimum size."""

from pathlib import Path

from agentic_rag.ingest.pipeline import read_chunks

chunks = read_chunks(Path("data/processed/chunks.jsonl"))
tiny = sorted((c for c in chunks if c.char_count < 100), key=lambda c: c.char_count)

print(f"chunks under 100 chars: {len(tiny)} of {len(chunks)}")
for c in tiny[:15]:
    print(f"{c.char_count:>4}  {c.chunk_id:<50}  {c.content[:60]!r}")
