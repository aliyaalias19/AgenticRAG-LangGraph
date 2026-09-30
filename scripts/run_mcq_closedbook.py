"""Closed-book MCQ evaluation: the question without its source passages.

The open-book variant hands the model the documentation and asks a question
about it, which measures reading comprehension -- a capability the base
instruct model already has. Fine-tuning changes what a model knows, not what
it can read, so an open-book score cannot detect it. Removing the passages
makes the question answerable only from weights, which is the thing QLoRA
actually moves.
"""

import argparse
import json

from agentic_rag.config.settings import get_settings
from agentic_rag.llm.provider import Message, OpenAICompatibleProvider
from agentic_rag.obs.logging import configure_logging
from agentic_rag.train.mcq import MCQ_FILENAME, OPTION_LABELS, read_items

SYSTEM = (
    "Answer the multiple-choice question about Kubernetes from your own "
    "knowledge. Reply with a single letter and nothing else."
)


def build_prompt(item) -> str:
    choices = "\n".join(f"{OPTION_LABELS[i]}. {option}" for i, option in enumerate(item.options))
    return f"Question: {item.question}\n\n{choices}\n\nAnswer:"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()

    items = read_items(settings.paths.evalsets_dir / MCQ_FILENAME)
    if args.limit:
        items = items[: args.limit]

    provider = OpenAICompatibleProvider(
        settings, base_url=args.base_url, model=args.model, api_key=""
    )

    correct = 0
    unparseable = 0
    per_item = []

    for item in items:
        reply = provider.complete(
            [Message("user", build_prompt(item))], system=SYSTEM, max_tokens=8
        )
        letter = next((c for c in reply.text.strip().upper() if c in OPTION_LABELS), None)
        if letter is None:
            unparseable += 1
            chosen = -1
        else:
            chosen = OPTION_LABELS.index(letter)
        hit = chosen == item.correct_index
        correct += int(hit)
        per_item.append(
            {
                "item_id": getattr(item, "item_id", None),
                "chosen": chosen,
                "correct_index": item.correct_index,
                "hit": hit,
            }
        )

    total = len(items)
    accuracy = correct / total if total else 0.0
    payload = {
        "label": args.label,
        "mode": "closed_book",
        "model": args.model,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "unparseable": unparseable,
        "per_item": per_item,
    }

    out = settings.paths.results_dir / f"mcq_closedbook_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"\n{args.label} (closed-book): {accuracy:.1%} ({correct}/{total}, {unparseable} unparseable)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
