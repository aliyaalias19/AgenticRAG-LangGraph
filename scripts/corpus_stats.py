"""Print corpus size distribution and section breakdown."""

import statistics as st
from collections import Counter
from pathlib import Path

from agentic_rag.ingest.pipeline import read_corpus

docs = read_corpus(Path("data/processed/corpus.jsonl"))
counts = sorted(d.char_count for d in docs)

print(f"documents : {len(docs)}")
print(f"p25       : {counts[len(counts) // 4]}")
print(f"median    : {st.median(counts)}")
print(f"p75       : {counts[3 * len(counts) // 4]}")
print(f"p95       : {counts[int(0.95 * len(counts))]}")
print(f"max       : {counts[-1]}")

print("\nsections:")
sections = Counter(d.section_path[0] if d.section_path else "(root)" for d in docs)
for section, n in sections.most_common():
    print(f"{n:>5}  {section}")
