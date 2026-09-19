"""Print corpus statistics broken down by language."""

import statistics as st
from collections import Counter
from pathlib import Path

from agentic_rag.ingest.pipeline import read_chunks, read_corpus

docs = read_corpus(Path("data/processed/corpus.jsonl"))
chunks = read_chunks(Path("data/processed/chunks.jsonl"))

print("documents by language:")
for lang, n in Counter(d.language for d in docs).most_common():
    print(f"  {lang}: {n}")

print("\nchunks by language:")
for lang, n in Counter(c.language for c in chunks).most_common():
    print(f"  {lang}: {n}")

for lang in ("en", "zh"):
    sizes = sorted(c.char_count for c in chunks if c.language == lang)
    if sizes:
        print(f"\n{lang} chunk chars — median {st.median(sizes)}, max {sizes[-1]}")

en_ids = {d.doc_id for d in docs if d.language == "en"}
zh_ids = {d.doc_id for d in docs if d.language == "zh"}
print(f"\nparallel documents (same doc_id in both): {len(en_ids & zh_ids)}")
print(f"english only: {len(en_ids - zh_ids)}")
print(f"chinese only: {len(zh_ids - en_ids)}")
