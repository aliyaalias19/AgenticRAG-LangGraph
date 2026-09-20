"""Generate questions from a handful of documents to inspect prompt quality."""

from pathlib import Path

from agentic_rag.config.settings import get_settings
from agentic_rag.eval.generator import assign_question_type, generate_for_document
from agentic_rag.eval.sampling import sample_documents
from agentic_rag.ingest.pipeline import read_corpus
from agentic_rag.llm.client import LLMClient
from agentic_rag.obs.logging import configure_logging

configure_logging()
settings = get_settings()

documents = read_corpus(Path("data/processed/corpus.jsonl"))
sampled = sample_documents(
    documents,
    total=10,
    section_weights=settings.eval.section_weights,
    min_chars=settings.eval.min_document_chars,
    max_chars=settings.eval.max_document_chars,
    seed=settings.random_seed,
)[:5]

client = LLMClient()

for index, document in enumerate(sampled):
    question_type = assign_question_type(index)
    questions = generate_for_document(
        document=document,
        question_type=question_type,
        count=2,
        client=client,
        max_chars=settings.eval.max_document_chars,
    )

    print(f"\n{'=' * 78}")
    print(f"doc   : {document.doc_id}")
    print(f"title : {document.title}")
    print(f"type  : {question_type}")
    print(f"chars : {document.char_count}")
    for question in questions:
        print(f"  Q: {question.question}")

print(f"\n{'=' * 78}")
print(f"usage: {client.usage.as_dict()}")
