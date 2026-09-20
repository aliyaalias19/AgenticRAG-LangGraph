"""Analyse the verified question set: rejections and lexical overlap."""

import statistics as st

from agentic_rag.config.settings import get_settings
from agentic_rag.eval.storage import VERIFIED_FILENAME, read_verified_set

settings = get_settings()
vs = read_verified_set(settings.paths.evalsets_dir / VERIFIED_FILENAME)

answerable = [q for q in vs.questions if q.is_answerable]
print(f"total      : {len(vs.questions)}")
print(f"answerable : {len(answerable)}")
print(f"dropped    : {len(vs.questions) - len(answerable)}")

print("\nrejection rate by question type:")
for qtype in ("direct", "paraphrased", "multi_hop", "scenario"):
    subset = [q for q in vs.questions if q.question_type == qtype]
    kept = sum(len(q.gold_chunk_ids) for q in subset)
    rejected = sum(len(q.rejected_chunk_ids) for q in subset)
    total = kept + rejected
    lost = sum(1 for q in subset if not q.is_answerable)
    if total:
        print(
            f"  {qtype:<14} {rejected}/{total} chunks rejected "
            f"({rejected / total:.0%}), {lost} questions dropped"
        )

print("\nlexical overlap by question type:")
for qtype in ("direct", "paraphrased", "multi_hop", "scenario"):
    subset = [q.lexical_overlap for q in answerable if q.question_type == qtype]
    if subset:
        print(
            f"  {qtype:<14} mean {st.mean(subset):.3f}  "
            f"median {st.median(subset):.3f}  n={len(subset)}"
        )

overlaps = sorted(q.lexical_overlap for q in answerable)
print("\noverall overlap distribution:")
print(f"  p10    {overlaps[len(overlaps) // 10]:.3f}")
print(f"  median {st.median(overlaps):.3f}")
print(f"  p90    {overlaps[9 * len(overlaps) // 10]:.3f}")

print("\nlowest-overlap questions (hardest for lexical retrieval):")
for q in sorted(answerable, key=lambda x: x.lexical_overlap)[:5]:
    print(f"  {q.lexical_overlap:.2f} [{q.question_type}] {q.question}")

print("\nquestions that lost all gold chunks:")
for q in vs.questions:
    if not q.is_answerable:
        print(f"  [{q.question_type}] {q.question[:90]}")
