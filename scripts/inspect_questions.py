"""Summarise a generated question set."""

from collections import Counter

from agentic_rag.config.settings import get_settings
from agentic_rag.eval.storage import QUESTIONS_FILENAME, read_question_set

settings = get_settings()
qs = read_question_set(settings.paths.evalsets_dir / QUESTIONS_FILENAME)

print(f"questions : {len(qs.questions)}")
print(f"model     : {qs.run.model}")
print(f"prompt    : {qs.run.prompt_version}")
print(f"tokens    : {qs.run.input_tokens} in, {qs.run.output_tokens} out")

print("\nby type:")
for qtype, n in Counter(q.question_type for q in qs.questions).most_common():
    print(f"  {qtype:<14} {n}")

print("\nby section:")
for section, n in Counter(q.source_section for q in qs.questions).most_common():
    print(f"  {section:<14} {n}")

for qtype in ("direct", "paraphrased", "multi_hop", "scenario"):
    print(f"\n--- {qtype} ---")
    for q in [x for x in qs.questions if x.question_type == qtype][:4]:
        print(f"  {q.question}")
        print(f"    from: {q.source_doc_id}")
