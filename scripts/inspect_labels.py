"""Inspect the labelled question set."""

from collections import Counter

from agentic_rag.config.settings import get_settings
from agentic_rag.eval.storage import LABELLED_FILENAME, read_labelled_set

settings = get_settings()
ls = read_labelled_set(settings.paths.evalsets_dir / LABELLED_FILENAME)

answerable = [q for q in ls.questions if q.is_answerable]
counts = Counter(len(q.gold_chunk_ids) for q in answerable)

print(f"total       : {len(ls.questions)}")
print(f"answerable  : {len(answerable)}")

print("\ngold chunks per question:")
for n in sorted(counts):
    print(f"  {n}: {counts[n]}")

print("\nmean gold chunks by question type:")
for qtype in ("direct", "paraphrased", "multi_hop", "scenario"):
    subset = [q for q in answerable if q.question_type == qtype]
    if subset:
        mean = sum(len(q.gold_chunk_ids) for q in subset) / len(subset)
        print(f"  {qtype:<14} {mean:.2f}  (n={len(subset)})")

print("\nunanswerable questions:")
for q in ls.questions:
    if not q.is_answerable:
        print(f"  [{q.question_type}] {q.question}")
        print(f"    from: {q.source_doc_id}")

print("\nsample labels:")
for q in answerable[:3]:
    print(f"\n  Q: {q.question}")
    print(f"  gold: {q.gold_chunk_ids}")
    print(f"  why : {q.labeller_reasoning}")
