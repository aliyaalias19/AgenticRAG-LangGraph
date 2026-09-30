"""Build the supervised fine-tuning dataset and the held-out MCQ set.

Training examples are generated from documents that the retrieval evaluation
does not draw on, and the held-out split is document-disjoint from training.
Both precautions exist for the same reason: a measured gain should come from
the model having learned the task, not from having seen the answer.

    python scripts/build_sft_dataset.py --examples 500 --mcq 200
"""

import argparse
import json
import random
import sys

from agentic_rag.config.settings import get_settings
from agentic_rag.eval.sampling import sample_documents
from agentic_rag.ingest.pipeline import (
    CHUNKS_FILENAME,
    CORPUS_FILENAME,
    read_chunks,
    read_corpus,
)
from agentic_rag.llm.parsing import ParseError, parse_json_object
from agentic_rag.llm.provider import Message, build_provider
from agentic_rag.obs.logging import configure_logging, get_logger
from agentic_rag.train.dataset import (
    ABSTENTION_RESPONSE,
    HELDOUT_FILENAME,
    TRAIN_FILENAME,
    InstructionExample,
    build_instruction,
    split_by_document,
    write_jsonl,
)
from agentic_rag.train.mcq import MCQ_FILENAME, MCQItem, write_items

logger = get_logger(__name__)

SFT_SYSTEM = """\
You write training examples for a Kubernetes documentation assistant.

Given a document, write a question an engineer would ask and the ideal
grounded answer to it. The answer must:
- use only what the document states
- cite passage numbers inline as [1], [2]
- be two to five sentences
- use the exact field names, flags and commands from the document

Return only valid JSON. No preamble, no markdown fences.\
"""

SFT_TEMPLATE = """\
Passages:
{passages}

Return JSON in exactly this shape:
{{"question": "...", "answer": "...", "question_type": "factual"}}\
"""

MCQ_SYSTEM = """\
You write multiple-choice questions that test understanding of Kubernetes
documentation.

Write one question with four options, exactly one correct. The three
distractors must be plausible to someone who half-remembers the material:
use real Kubernetes concepts that are wrong in this specific context, not
obvious nonsense. A distractor nobody would pick tests nothing.

Return only valid JSON. No preamble, no markdown fences.\
"""

MCQ_TEMPLATE = """\
Passages:
{passages}

Return JSON in exactly this shape:
{{"question": "...", "options": ["a", "b", "c", "d"], "correct_index": 0}}\
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--mcq", type=int, default=200)
    parser.add_argument("--heldout-fraction", type=float, default=0.15)
    parser.add_argument("--abstention-fraction", type=float, default=0.10)
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    rng = random.Random(settings.random_seed)  # noqa: S311 - reproducibility

    documents = read_corpus(settings.paths.processed_dir / CORPUS_FILENAME)
    chunks = read_chunks(settings.paths.processed_dir / CHUNKS_FILENAME)
    by_document: dict[str, list] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.doc_id, []).append(chunk)

    # Sample both languages. The assistant is bilingual, so drawing training
    # data from English alone teaches it half the task -- and the English pool
    # is not large enough to fill the requested dataset on its own.
    per_language = (args.examples + args.mcq) // 2 + 1
    sampled = []
    for offset, language in ((1, "en"), (2, "zh")):
        sampled.extend(
            sample_documents(
                documents,
                total=per_language,
                section_weights=settings.eval.section_weights,
                min_chars=settings.eval.min_document_chars,
                max_chars=settings.eval.max_document_chars,
                seed=settings.random_seed + offset,
                language=language,
            )
        )
    provider = build_provider(settings)

    examples: list[InstructionExample] = []
    mcq_items: list[MCQItem] = []

    for index, document in enumerate(sampled):
        doc_chunks = sorted(by_document.get(document.doc_id, []), key=lambda c: c.chunk_index)[:4]
        if not doc_chunks:
            continue
        passages = [c.content for c in doc_chunks]
        numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))

        building_mcq = len(mcq_items) < args.mcq and index % 3 == 2

        try:
            if building_mcq:
                response = provider.complete(
                    [Message("user", MCQ_TEMPLATE.format(passages=numbered))],
                    system=MCQ_SYSTEM,
                )
                parsed = parse_json_object(response.text)
                options = [str(o) for o in parsed.get("options", [])]
                correct = int(parsed.get("correct_index", -1))
                if len(options) == 4 and 0 <= correct < 4:
                    mcq_items.append(
                        MCQItem(
                            item_id=f"mcq-{len(mcq_items):04d}",
                            question=str(parsed["question"]),
                            options=options,
                            correct_index=correct,
                            passages=passages,
                            source_doc_id=document.doc_id,
                        )
                    )
            elif len(examples) < args.examples:
                response = provider.complete(
                    [Message("user", SFT_TEMPLATE.format(passages=numbered))],
                    system=SFT_SYSTEM,
                )
                parsed = parse_json_object(response.text)
                examples.append(
                    InstructionExample(
                        example_id=f"sft-{len(examples):04d}",
                        instruction=build_instruction(str(parsed["question"]), passages),
                        response=str(parsed["answer"]),
                        source_doc_id=document.doc_id,
                        question_type=str(parsed.get("question_type", "factual")),
                    )
                )
        except (ParseError, KeyError, ValueError) as error:
            logger.warning(
                "example_generation_failed",
                doc_id=document.doc_id,
                error=str(error),
            )

        if (index + 1) % 50 == 0:
            logger.info(
                "generation_progress",
                done=index + 1,
                examples=len(examples),
                mcq=len(mcq_items),
            )

    # Abstention examples teach the model to decline when the passages do not
    # support an answer. Without them the model learns that every prompt has
    # an answer in it, which is exactly the habit the evidence gate has to
    # fight at inference time.
    abstention_count = int(len(examples) * args.abstention_fraction)
    for i in range(abstention_count):
        document = sampled[rng.randrange(len(sampled))]
        unrelated = sampled[rng.randrange(len(sampled))]
        doc_chunks = by_document.get(unrelated.doc_id, [])[:3]
        if not doc_chunks:
            continue
        examples.append(
            InstructionExample(
                example_id=f"sft-abstain-{i:04d}",
                instruction=build_instruction(
                    f"What does the documentation say about {document.title}?",
                    [c.content for c in doc_chunks],
                ),
                response=ABSTENTION_RESPONSE,
                source_doc_id=unrelated.doc_id,
                question_type="abstention",
                is_abstention=True,
            )
        )

    split = split_by_document(examples, args.heldout_fraction, settings.random_seed)
    write_jsonl(split.train, settings.paths.evalsets_dir / TRAIN_FILENAME)
    write_jsonl(split.heldout, settings.paths.evalsets_dir / HELDOUT_FILENAME)
    write_items(mcq_items, settings.paths.evalsets_dir / MCQ_FILENAME)

    print(json.dumps(split.summary(), indent=2))
    print(f"mcq_items: {len(mcq_items)}")
    print(f"usage: {provider.usage.as_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
