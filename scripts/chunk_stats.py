"""Print chunk size distribution and structural statistics."""

import statistics as st
from collections import Counter
from pathlib import Path

from agentic_rag.ingest.pipeline import read_chunks

chunks = read_chunks(Path("data/processed/chunks.jsonl"))
counts = sorted(c.char_count for c in chunks)

print(f"chunks       : {len(chunks)}")
print(f"min          : {counts[0]}")
print(f"p25          : {counts[len(counts) // 4]}")
print(f"median       : {st.median(counts)}")
print(f"p75          : {counts[3 * len(counts) // 4]}")
print(f"max          : {counts[-1]}")
print(f"mean         : {sum(counts) // len(counts)}")

per_doc = Counter(c.doc_id for c in chunks)
doc_counts = sorted(per_doc.values())
print(f"\nchunks/doc median : {st.median(doc_counts)}")
print(f"chunks/doc max    : {doc_counts[-1]}")

no_heading = sum(1 for c in chunks if not c.heading_path)
print(f"\nchunks without heading context: {no_heading} ({no_heading / len(chunks):.1%})")

print("\nsample chunk:")
sample = next(c for c in chunks if c.heading_path and 400 < c.char_count < 900)
print(f"  id       : {sample.chunk_id}")
print(f"  headings : {' > '.join(sample.heading_path)}")
print(f"  preview  : {sample.content[:200]}...")
